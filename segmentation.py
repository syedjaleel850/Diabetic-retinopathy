# app.py
import streamlit as st
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm.auto import tqdm
import warnings
import itertools
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Essential Configurations (MUST MATCH YOUR TRAINED MODEL) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_INPUT_PATCH_SIZE = 256
INFERENCE_STRIDE = 128
ENCODER_NAME = 'swinv2_small_window16_256' # Renamed to avoid conflict with ENCODER global in some environments

CLASS_NAMES = ['MA', 'HE', 'EX', 'SE', 'OD']
NUM_CLASSES = len(CLASS_NAMES)

_available_colors_list = ['red', 'lime', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'blue', 'pink', 'brown']
VIS_COLORS = list(itertools.islice(itertools.cycle(_available_colors_list), NUM_CLASSES))
PREDICTION_THRESHOLD = 0.5

_NORM_MEAN = (0.485, 0.456, 0.406)
_NORM_STD = (0.229, 0.224, 0.225)

# --- Model Definition (Copied & Adapted from your multi-lesion script) ---
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1,1,0,bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1,1,0,bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int,1,1,1,0,bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1, x1 = self.W_g(g), self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            x1 = F.interpolate(x1, size=g1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1+x1); psi = self.psi(psi); return x * psi

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super().__init__(); self.use_attention = use_attention
        up_out_channels = in_channels // 2; combined_channels = up_out_channels + skip_channels
        self.up = nn.ConvTranspose2d(in_channels, up_out_channels, kernel_size=2, stride=2)
        if self.use_attention: self.attention_gate = AttentionGate(F_g=up_out_channels, F_l=skip_channels, F_int=skip_channels//2)
        else: self.attention_gate = None
        self.conv_block = nn.Sequential(nn.Conv2d(combined_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x, skip=None):
        x_up = self.up(x)
        if skip is not None:
            skip_aligned = F.interpolate(skip, size=x_up.shape[2:], mode='bilinear', align_corners=False) if x_up.shape[2:] != skip.shape[2:] else skip
            skip_final = self.attention_gate(g=x_up, x=skip_aligned) if self.use_attention and self.attention_gate else skip_aligned
            x = torch.cat([x_up, skip_final], dim=1)
        else: x = x_up
        return self.conv_block(x)

class CustomSwinUNet(nn.Module):
    def __init__(self, encoder=ENCODER_NAME, encoder_weights=None, classes=NUM_CLASSES,
                 decoder_channels=(256, 128, 64, 32), use_attention_gates=True,
                 img_size=MODEL_INPUT_PATCH_SIZE, debug_prints=False): # Added img_size
        super().__init__(); self.classes = classes; self.debug_prints = debug_prints; self.use_attention_gates = use_attention_gates
        if len(decoder_channels) != 4: raise ValueError("decoder_channels must have 4 values.")
        
        self.encoder = timm.create_model(
            encoder, pretrained=(encoder_weights == 'imagenet'), features_only=True, img_size=img_size,
            pretrained_window_sizes=(0,0,0,0) if 'swin' in encoder else None 
        )
        encoder_out_channels = self.encoder.feature_info.channels()
        if self.debug_prints: print(f"Encoder feature channels: {encoder_out_channels}")

        if len(encoder_out_channels) < 4:
            raise ValueError(f"Encoder provides {len(encoder_out_channels)} feature levels. Need 4.")

        bottleneck_channels = encoder_out_channels[-1]
        skip_channels_for_decoder = encoder_out_channels[:-1][::-1] 

        self.decoder1 = DecoderBlock(bottleneck_channels, skip_channels_for_decoder[0], decoder_channels[0], use_attention=self.use_attention_gates)
        self.decoder2 = DecoderBlock(decoder_channels[0], skip_channels_for_decoder[1], decoder_channels[1], use_attention=self.use_attention_gates)
        self.decoder3 = DecoderBlock(decoder_channels[1], skip_channels_for_decoder[2], decoder_channels[2], use_attention=self.use_attention_gates)
        self.decoder4 = nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2)
        self.final_relu = nn.ReLU(inplace=True)
        self.segmentation_head = nn.Conv2d(decoder_channels[3], self.classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape
        features = self.encoder(x)
        if features[-1].ndim == 4 and self.encoder.feature_info and \
           self.encoder.feature_info.channels()[-1] != features[-1].shape[1] and \
           self.encoder.feature_info.channels()[-1] == features[-1].shape[-1]:
            features = [f.permute(0, 3, 1, 2).contiguous() for f in features]

        bottleneck_feature = features[-1]
        skip_features = features[:-1][::-1]

        d1 = self.decoder1(bottleneck_feature, skip_features[0])
        d2 = self.decoder2(d1, skip_features[1])
        d3 = self.decoder3(d2, skip_features[2])
        d4 = self.final_relu(self.decoder4(d3))
        logits = self.segmentation_head(d4)
        if logits.shape[2:] != input_shape[2:]:
            logits = F.interpolate(logits, size=input_shape[2:], mode='bilinear', align_corners=False)
        return logits

@st.cache_resource # Cache the model loading
def load_model(model_path, encoder_name, num_classes, model_input_size):
    # print(f"Building model for Streamlit: Encoder='{encoder_name}', Classes={num_classes}, ImgSize={model_input_size}")
    model = CustomSwinUNet(encoder=encoder_name, classes=num_classes, img_size=model_input_size, use_attention_gates=True)
    model.to(DEVICE)
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success(f"Loaded model state_dict from: {os.path.basename(model_path)}")
        else:
            model.load_state_dict(checkpoint)
            st.success(f"Loaded raw state_dict from: {os.path.basename(model_path)}")
    except Exception as e:
        st.error(f"Error loading model weights from {model_path}: {e}")
        return None
    model.eval()
    return model

# --- Helper Functions for Prediction and Visualization ---
def get_val_transforms_st(img_size=MODEL_INPUT_PATCH_SIZE):
    return A.Compose([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
        A.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
        ToTensorV2(),
    ])

def load_original_image_st(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image) # Convert to NumPy array (RGB)

def predict_patch_probabilities_multiclass_st(model, image_patch_rgb, device, transforms):
    model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        augmented = transforms(image=image_patch_rgb)
        image_tensor = augmented['image'].unsqueeze(0).to(device)
        logits = model(image_tensor)
        pred_probs = torch.sigmoid(logits)
        return pred_probs.squeeze(0).cpu().numpy()

def predict_full_image_patchwise_multiclass_st(model, full_image_rgb, patch_size, stride, device, transforms, num_classes, threshold=PREDICTION_THRESHOLD):
    img_h, img_w = full_image_rgb.shape[:2]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Processing image of size {img_w}x{img_h} with {patch_size}x{patch_size} patches...")

    if img_h < patch_size or img_w < patch_size:
        status_text.text(f"Image smaller than patch size. Resizing for single prediction.")
        resized_img = cv2.resize(full_image_rgb, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        probs_map_patch_size = predict_patch_probabilities_multiclass_st(model, resized_img, device, transforms)
        final_probs_upscaled = np.zeros((num_classes, img_h, img_w), dtype=np.float32)
        for c in range(num_classes):
            final_probs_upscaled[c, :, :] = cv2.resize(probs_map_patch_size[c,:,:], (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        final_binary_mask_multiclass = (final_probs_upscaled > threshold).astype(np.uint8)
        progress_bar.progress(100)
        status_text.text("Prediction complete!")
        return final_binary_mask_multiclass, final_probs_upscaled

    stitched_probabilities = np.zeros((num_classes, img_h, img_w), dtype=np.float32)
    counts = np.zeros((img_h, img_w), dtype=np.float32)
    
    patches_info = []
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            y_end = min(y + patch_size, img_h)
            x_end = min(x + patch_size, img_w)
            patches_info.append((y, x, y_end, x_end))
            
    total_patches = len(patches_info)
    for idx, (y_start, x_start, y_end, x_end) in enumerate(patches_info):
        patch_from_img = full_image_rgb[y_start:y_end, x_start:x_end, :]
        actual_patch_h, actual_patch_w = patch_from_img.shape[:2]
        if actual_patch_h == 0 or actual_patch_w == 0: continue

        patch_for_model = cv2.resize(patch_from_img, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        patch_probs_model_size = predict_patch_probabilities_multiclass_st(model, patch_for_model, device, transforms)
        
        for c in range(num_classes):
            class_probs_actual_size = cv2.resize(patch_probs_model_size[c,:,:], (actual_patch_w, actual_patch_h), interpolation=cv2.INTER_LINEAR)
            stitched_probabilities[c, y_start:y_end, x_start:x_end] += class_probs_actual_size
        counts[y_start:y_end, x_start:x_end] += 1
        
        progress_bar.progress((idx + 1) / total_patches)
        status_text.text(f"Predicting patch {idx+1}/{total_patches}...")

    averaged_probabilities_multiclass = np.zeros((num_classes, img_h, img_w), dtype=np.float32)
    for c in range(num_classes):
        averaged_probabilities_multiclass[c,:,:] = np.where(counts > 0, stitched_probabilities[c,:,:] / counts, 0)
    final_binary_mask_multiclass = (averaged_probabilities_multiclass > threshold).astype(np.uint8)
    
    status_text.text("Prediction complete!")
    return final_binary_mask_multiclass, averaged_probabilities_multiclass

def create_prediction_overlay_st(original_image_rgb, binary_mask, color_str='yellow', alpha=0.6):
    h, w = binary_mask.shape
    if original_image_rgb.dtype != np.uint8:
        if np.max(original_image_rgb) <= 1.0 and (original_image_rgb.dtype == np.float32 or original_image_rgb.dtype == np.float64):
            overlay_img_base = (original_image_rgb * 255).astype(np.uint8)
        else: overlay_img_base = np.clip(original_image_rgb, 0, 255).astype(np.uint8)
    else: overlay_img_base = original_image_rgb.copy()
    if overlay_img_base.shape[:2] != (h,w):
        overlay_img_base = cv2.resize(overlay_img_base, (w,h), interpolation=cv2.INTER_LINEAR)
    color_rgb_array = np.array(mcolors.to_rgb(color_str)) * 255
    mask_colored_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_colored_rgb[binary_mask == 1] = color_rgb_array
    return cv2.addWeighted(mask_colored_rgb, alpha, overlay_img_base, 1 - alpha, 0)

def create_multiclass_segmentation_figure(original_img_rgb, pred_masks_stack, prob_maps_stack, class_names_list, vis_colors_list):
    num_classes_to_disp = len(class_names_list)
    
    # For each class: Original, Overlay, Binary Mask, Probability Map
    fig, axes = plt.subplots(num_classes_to_disp, 4, figsize=(20, 4.5 * num_classes_to_disp)) # Adjusted figsize
    if num_classes_to_disp == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle("Multi-Lesion Segmentation Results", fontsize=18, y=1.0) # Adjusted y for suptitle

    for i in range(num_classes_to_disp):
        class_name = class_names_list[i]
        color_str = vis_colors_list[i]

        # Original Image
        axes[i, 0].imshow(original_img_rgb)
        axes[i, 0].set_title(f"Input (for {class_name})", fontsize=10)
        axes[i, 0].axis('off')

        # Predicted Overlay for this class
        pred_overlay = create_prediction_overlay_st(original_img_rgb, pred_masks_stack[i, :, :], color_str=color_str)
        axes[i, 1].imshow(pred_overlay)
        axes[i, 1].set_title(f"Predicted Overlay: {class_name}", fontsize=10)
        axes[i, 1].axis('off')
        
        # Predicted Binary Mask for this class
        axes[i, 2].imshow(pred_masks_stack[i, :, :], cmap='gray')
        axes[i, 2].set_title(f"Predicted Binary: {class_name}", fontsize=10)
        axes[i, 2].axis('off')
        
        # Predicted Probability map for this class
        prob_map_display = prob_maps_stack[i, :, :]
        im = axes[i, 3].imshow(prob_map_display, cmap='viridis', vmin=0, vmax=1)
        axes[i, 3].set_title(f"Probability Map: {class_name}", fontsize=10)
        axes[i, 3].axis('off')
        # fig.colorbar(im, ax=axes[i,3], fraction=0.046, pad=0.04) # Can make plot too busy

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust rect for suptitle
    return fig

# --- Streamlit App ---
def main():
    st.set_page_config(layout="wide", page_title="Multi-Lesion Retinal Segmentation")
    st.title("👁️ Multi-Lesion Retinal Segmentation App")
    st.markdown("""
    Upload a retinal fundus image to perform multi-lesion segmentation.
    The model will predict masks for: **MA, HE, EX, SE, OD**.
    """)

    # --- Model Path Configuration ---
    # IMPORTANT: Place your trained .pth model file in the same directory as this app.py,
    # or provide the full path.
    DEFAULT_MODEL_PATH = "Fine tune model/idrid_multi_lesion_v1_256_attn_swin_s_w16_patch_finetuned_run1_final_model.pth" 
    
    model_path = st.sidebar.text_input("Enter Path to Model (.pth file):", DEFAULT_MODEL_PATH)
    
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model not found at: {model_path}. Please ensure the path is correct.")
        st.stop()

    # Load Model
    model = load_model(model_path, ENCODER_NAME, NUM_CLASSES, MODEL_INPUT_PATCH_SIZE)
    if model is None:
        st.error("Model could not be loaded. Please check the path and model file.")
        st.stop()

    # --- Image Upload ---
    uploaded_file = st.file_uploader("Choose a retinal fundus image...", type=["jpg", "jpeg", "png", "tif", "bmp"])

    if uploaded_file is not None:
        original_image_rgb = load_original_image_st(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            st.image(original_image_rgb, caption="Original Uploaded Image", use_column_width=True)

        if st.button("🔍 Segment Lesions"):
            with st.spinner("Segmenting lesions... This may take a moment for large images."):
                patch_transforms = get_val_transforms_st(img_size=MODEL_INPUT_PATCH_SIZE)
                
                pred_binary_masks_stack, pred_probs_stack = predict_full_image_patchwise_multiclass_st(
                    model, 
                    original_image_rgb, 
                    patch_size=MODEL_INPUT_PATCH_SIZE, 
                    stride=INFERENCE_STRIDE, 
                    device=DEVICE, 
                    transforms=patch_transforms,
                    num_classes=NUM_CLASSES,
                    threshold=PREDICTION_THRESHOLD
                )
            
            st.success("Segmentation Complete!")
            
            # Display results using Matplotlib figure in Streamlit
            st.subheader("Segmentation Results")
            fig = create_multiclass_segmentation_figure(
                original_image_rgb,
                pred_binary_masks_stack, # We don't have GT in testing app, so pass dummy or skip
                pred_probs_stack,
                CLASS_NAMES,
                VIS_COLORS
            )
            st.pyplot(fig)
            
            # Optionally, save individual predicted masks
            # save_individual_masks = st.checkbox("Save individual predicted masks?")
            # if save_individual_masks:
            #     output_dir_masks = "predicted_masks_output"
            #     os.makedirs(output_dir_masks, exist_ok=True)
            #     img_basename = os.path.splitext(uploaded_file.name)[0]
            #     for i, class_name in enumerate(CLASS_NAMES):
            #         mask_to_save = (pred_binary_masks_stack[i] * 255).astype(np.uint8)
            #         save_path = os.path.join(output_dir_masks, f"{img_basename}_pred_{class_name}.png")
            #         cv2.imwrite(save_path, mask_to_save)
            #     st.info(f"Individual predicted masks saved to '{output_dir_masks}' directory.")

    else:
        st.info("Please upload an image to get started.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model & Prediction Info")
    st.sidebar.markdown(f"**Encoder:** `{ENCODER_NAME}`")
    st.sidebar.markdown(f"**Model Patch Size:** `{MODEL_INPUT_PATCH_SIZE}x{MODEL_INPUT_PATCH_SIZE}`")
    st.sidebar.markdown(f"**Inference Stride:** `{INFERENCE_STRIDE}`")
    st.sidebar.markdown(f"**Classes:** `{', '.join(CLASS_NAMES)}`")
    st.sidebar.markdown(f"**Device:** `{DEVICE}`")

if __name__ == '__main__':
    main()
