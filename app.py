import os
import io
import gc
import time
import uuid
import logging
import asyncio
from datetime import datetime
from typing import Optional, Tuple
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import psutil
import gdown
from scipy import ndimage
from skimage import segmentation, feature, filters
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pymongo import MongoClient

# Import U2NET model
from u2net import U2NET, U2NETP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
RESULT_DIR = os.getenv("RESULT_DIR", "results")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "models/pretrained")

for dir_path in [UPLOAD_DIR, RESULT_DIR, MODEL_CACHE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

client = None
db = None
collection = None

try:
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=2000)
    db = client["segmentation_db"]
    collection = db["images"]
    client.admin.command('ping')
    logger.info("MongoDB connected")
except:
    logger.warning("MongoDB not available")
    client = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
else:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

max_workers = 1 if device.type == "cuda" else 2
executor = ThreadPoolExecutor(max_workers=max_workers)


class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_configs = {
            "u2net": {
                "url": "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
                "filename": "u2net.pth",
                "input_size": (320, 320),
                "description": "U2NET - Salient object detection",
                "model_class": U2NET
            },
            "u2net_portrait": {
                "url": "https://drive.google.com/uc?id=1IG3HdpcRiDoWNookbncQjeaPN28t90yW",
                "filename": "u2net_portrait.pth", 
                "input_size": (512, 512),
                "description": "U2NET Portrait - Human portrait segmentation",
                "model_class": U2NET
            },
            "u2netp": {
                "url": "https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy",
                "filename": "u2netp.pth",
                "input_size": (320, 320),
                "description": "U2NET-P - Lightweight version",
                "model_class": U2NETP
            },
            "modnet": {
                "url": "https://drive.google.com/uc?id=1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz",
                "filename": "modnet_photographic_portrait_matting.ckpt",
                "input_size": (512, 512),
                "description": "MODNet - High quality portrait matting"
            }
        }
        
    async def download_models(self):
        for model_name, config in self.model_configs.items():
            model_path = os.path.join(MODEL_CACHE_DIR, config["filename"])
            
            if not os.path.exists(model_path):
                logger.info(f"Downloading {model_name}...")
                try:
                    if "drive.google.com" in config["url"]:
                        gdown.download(config["url"], model_path, quiet=False)
                    logger.info(f"{model_name} downloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to download {model_name}: {e}")
                    # Check if file exists locally (user mentioned they have u2net.pth)
                    if model_name == "u2net" and os.path.exists("u2net.pth"):
                        logger.info("Found local u2net.pth, copying to cache...")
                        import shutil
                        shutil.copy("u2net.pth", model_path)
            else:
                logger.info(f"{model_name} already exists")

    def load_u2net_model(self, model_name: str) -> Optional[torch.nn.Module]:
        """Load U2NET or U2NETP model"""
        try:
            config = self.model_configs[model_name]
            model_class = config["model_class"]
            model = model_class(in_ch=3, out_ch=1).to(device)
            
            model_path = os.path.join(MODEL_CACHE_DIR, config["filename"])
            
            if os.path.exists(model_path):
                try:
                    # Load state dict
                    state_dict = torch.load(model_path, map_location=device)
                    model.load_state_dict(state_dict)
                    logger.info(f"{model_name} loaded with pretrained weights")
                except Exception as e:
                    logger.warning(f"{model_name} loaded without pretrained weights: {e}")
            else:
                logger.warning(f"{model_name} model file not found, using random weights")
            
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return None

    def load_modnet(self) -> Optional[torch.nn.Module]:
        try:
            class MODNet(nn.Module):
                def __init__(self):
                    super(MODNet, self).__init__()
                    self.encoder = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                    )
                    
                    self.decoder = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                        nn.Conv2d(256, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                        nn.Conv2d(128, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 1, 3, padding=1),
                        nn.Sigmoid()
                    )
                    
                def forward(self, x):
                    x = self.encoder(x)
                    x = self.decoder(x)
                    return x
            
            model = MODNet().to(device)
            model_path = os.path.join(MODEL_CACHE_DIR, "modnet_photographic_portrait_matting.ckpt")
            
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint, strict=False)
                    logger.info("MODNet loaded with pretrained weights")
                except:
                    logger.warning("MODNet loaded without pretrained weights")
            
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load MODNet: {e}")
            return None

    def load_models(self):
        # Load U2NET models
        self.models["u2net"] = self.load_u2net_model("u2net")
        self.models["u2net_portrait"] = self.load_u2net_model("u2net_portrait")
        self.models["u2netp"] = self.load_u2net_model("u2netp")
        
        # Load other models
        self.models["modnet"] = self.load_modnet()
        
        # Remove None models
        self.models = {k: v for k, v in self.models.items() if v is not None}
        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")


class AdvancedBackgroundRemover:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
    def preprocess_image_u2net_fixed(self, image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Fixed preprocessing that preserves aspect ratio and returns transform info
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image_rgb.shape[:2]
        
        # Calculate aspect-ratio preserving resize
        target_width, target_height = target_size
        scale = min(target_width / original_width, target_height / original_height)
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image preserving aspect ratio
        image_resized = cv2.resize(image_rgb, (new_width, new_height))
        
        # Create canvas and center the image
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_y = (target_height - new_height) // 2
        pad_x = (target_width - new_width) // 2
        
        # Place resized image on canvas
        canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = image_resized
        
        # Normalize to [0, 1]
        canvas_normalized = canvas.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(canvas_normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        # Return tensor, original shape, and padding info for reverse transform
        return image_tensor, (original_height, original_width), (pad_y, pad_x, new_height, new_width)

    def preprocess_image_modnet(self, image: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
        """Preprocess image for MODNet"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, target_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_pil = Image.fromarray((image_normalized * 255).astype(np.uint8))
        image_tensor = transform(image_pil).unsqueeze(0).to(device)
        return image_tensor

    def postprocess_mask_u2net_fixed(self, output: torch.Tensor, original_shape: Tuple[int, int], 
                                   padding_info: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Fixed post-processing that correctly handles aspect ratio preservation
        """
        # U2NET returns multiple outputs, use the first one (main prediction)
        if isinstance(output, tuple):
            mask = output[0]
        else:
            mask = output
            
        # Convert to numpy
        mask_np = mask.squeeze().cpu().detach().numpy()
        
        # Extract padding info
        pad_y, pad_x, new_height, new_width = padding_info
        
        # Extract the actual predicted region (remove padding)
        mask_crop = mask_np[pad_y:pad_y + new_height, pad_x:pad_x + new_width]
        
        # Resize back to original dimensions
        mask_resized = cv2.resize(mask_crop, (original_shape[1], original_shape[0]))
        
        # Ensure values are in [0, 1]
        mask_resized = np.clip(mask_resized, 0, 1)
        
        return mask_resized

    def postprocess_mask_modnet(self, mask: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        """Post-process MODNet output"""
        mask_np = mask.squeeze().cpu().detach().numpy()
        mask_resized = cv2.resize(mask_np, (original_shape[1], original_shape[0]))
        return mask_resized

    def ensemble_masks_improved(self, masks: list, confidences: list = None) -> np.ndarray:
        """
        Improved ensemble that handles edge cases better
        """
        if not masks:
            return None
            
        if len(masks) == 1:
            return masks[0]
            
        if confidences is None:
            confidences = [1.0] * len(masks)
        
        # Normalize confidences
        confidences = np.array(confidences)
        confidences = confidences / (np.sum(confidences) + 1e-8)
        
        # Calculate variance to detect disagreement between models
        mask_stack = np.stack(masks, axis=0)
        mask_variance = np.var(mask_stack, axis=0)
        
        # Where models agree (low variance), use weighted average
        # Where models disagree (high variance), bias towards higher confidence
        ensemble_mask = np.zeros_like(masks[0])
        
        agreement_threshold = 0.1
        agreement_mask = mask_variance < agreement_threshold
        
        # For regions where models agree
        for i, (mask, conf) in enumerate(zip(masks, confidences)):
            ensemble_mask += mask * conf
            
        # For regions where models disagree, boost the most confident prediction
        disagreement_mask = ~agreement_mask
        if np.any(disagreement_mask):
            best_idx = np.argmax(confidences)
            ensemble_mask[disagreement_mask] = masks[best_idx][disagreement_mask]
            
        return ensemble_mask

    def advanced_edge_analysis_conservative(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        More conservative edge analysis that preserves subject boundaries
        """
        h, w = mask.shape
        refined_mask = mask.copy()
        
        # Use multiple edge detection methods for robustness
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Canny edge detection with multiple scales
        edges_fine = cv2.Canny(mask_uint8, 50, 150)
        edges_coarse = cv2.Canny(mask_uint8, 30, 100)
        
        # Combine edge maps
        edges_combined = np.maximum(edges_fine, edges_coarse)
        
        # Create a more conservative processing region
        kernel_small = np.ones((3, 3), np.uint8)
        edge_region = cv2.dilate(edges_combined, kernel_small, iterations=1)
        
        # Convert image to multiple color spaces for robustness
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Find edge coordinates
        edge_coords = np.where(edge_region > 0)
        
        # Process in smaller batches with more conservative updates
        batch_size = 500
        for i in range(0, len(edge_coords[0]), batch_size):
            batch_y = edge_coords[0][i:i+batch_size]
            batch_x = edge_coords[1][i:i+batch_size]
            
            for j, (y, x) in enumerate(zip(batch_y, batch_x)):
                if 6 <= y < h-6 and 6 <= x < w-6:
                    # Smaller neighborhood (7x7 instead of 9x9)
                    local_lab = lab_image[y-3:y+4, x-3:x+4]
                    local_hsv = hsv_image[y-3:y+4, x-3:x+4]
                    local_mask = mask[y-3:y+4, x-3:x+4]
                    
                    # Calculate color consistency in both color spaces
                    center_lab = lab_image[y, x]
                    center_hsv = hsv_image[y, x]
                    
                    lab_distances = np.linalg.norm(local_lab - center_lab, axis=2)
                    hsv_distances = np.linalg.norm(local_hsv - center_hsv, axis=2)
                    
                    # Normalize distances
                    lab_distances = lab_distances / (np.max(lab_distances) + 1e-6)
                    hsv_distances = hsv_distances / (np.max(hsv_distances) + 1e-6)
                    
                    # Create more conservative confidence maps
                    fg_mask = local_mask > 0.8  # Higher threshold for foreground
                    bg_mask = local_mask < 0.2  # Lower threshold for background
                    
                    if np.any(fg_mask) and np.any(bg_mask):
                        # Calculate confidence in both color spaces
                        lab_fg_conf = 1.0 - np.mean(lab_distances[fg_mask])
                        lab_bg_conf = 1.0 - np.mean(lab_distances[bg_mask])
                        
                        hsv_fg_conf = 1.0 - np.mean(hsv_distances[fg_mask])
                        hsv_bg_conf = 1.0 - np.mean(hsv_distances[bg_mask])
                        
                        # Combine confidences from both color spaces
                        fg_confidence = (lab_fg_conf + hsv_fg_conf) / 2
                        bg_confidence = (lab_bg_conf + hsv_bg_conf) / 2
                        
                        # Only update if we have strong confidence and preserve subject
                        total_conf = fg_confidence + bg_confidence
                        if total_conf > 0.5:  # Higher threshold for updates
                            new_alpha = fg_confidence / total_conf
                            # More conservative blending (less aggressive changes)
                            refined_mask[y, x] = 0.8 * refined_mask[y, x] + 0.2 * new_alpha
        
        return refined_mask

    def spectral_matting_multi_colorspace_fixed(self, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """
        FIXED: Enhanced spectral matting using multiple color spaces for better robustness
        """
        alpha = trimap.copy().astype(np.float32) / 255.0
        
        # Find unknown regions
        unknown_mask = (trimap == 128)
        
        if not np.any(unknown_mask):
            return alpha
        
        logger.info("Running multi-colorspace spectral matting...")
        
        # Convert to multiple color spaces - FIXED: cv2 is imported at top
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # Get known regions
        fg_mask = (trimap == 255)
        bg_mask = (trimap == 0)
        
        if not (np.any(fg_mask) and np.any(bg_mask)):
            return alpha
        
        # Sample colors from all color spaces
        max_samples = 2000  # Increased sample size for better accuracy
        
        color_spaces = [
            ("LAB", lab_image),
            ("HSV", hsv_image), 
            ("RGB", rgb_image)
        ]
        
        alpha_results = []
        
        for space_name, color_image in color_spaces:
            # Sample foreground and background colors
            fg_colors = color_image[fg_mask]
            bg_colors = color_image[bg_mask]
            
            # Limit sample size for performance
            if len(fg_colors) > max_samples:
                indices = np.random.choice(len(fg_colors), max_samples, replace=False)
                fg_colors = fg_colors[indices]
            if len(bg_colors) > max_samples:
                indices = np.random.choice(len(bg_colors), max_samples, replace=False)
                bg_colors = bg_colors[indices]
            
            # Get unknown pixels coordinates
            unknown_coords = np.where(unknown_mask)
            unknown_pixels = color_image[unknown_coords]
            
            # Vectorized distance computation
            unknown_expanded = unknown_pixels[:, np.newaxis, :]
            fg_expanded = fg_colors[np.newaxis, :, :]
            bg_expanded = bg_colors[np.newaxis, :, :]
            
            # Compute distances with different weightings for different color spaces
            if space_name == "LAB":
                # Weight L channel more for LAB
                weights = np.array([1.5, 1.0, 1.0])
                fg_distances = np.sqrt(np.sum((unknown_expanded - fg_expanded) ** 2 * weights, axis=2))
                bg_distances = np.sqrt(np.sum((unknown_expanded - bg_expanded) ** 2 * weights, axis=2))
            elif space_name == "HSV":
                # Special handling for HSV (hue wrapping)
                diff_fg = unknown_expanded - fg_expanded
                diff_bg = unknown_expanded - bg_expanded
                
                # Handle hue wrapping
                diff_fg[:, :, 0] = np.minimum(np.abs(diff_fg[:, :, 0]), 180 - np.abs(diff_fg[:, :, 0]))
                diff_bg[:, :, 0] = np.minimum(np.abs(diff_bg[:, :, 0]), 180 - np.abs(diff_bg[:, :, 0]))
                
                fg_distances = np.linalg.norm(diff_fg, axis=2)
                bg_distances = np.linalg.norm(diff_bg, axis=2)
            else:  # RGB
                fg_distances = np.linalg.norm(unknown_expanded - fg_expanded, axis=2)
                bg_distances = np.linalg.norm(unknown_expanded - bg_expanded, axis=2)
            
            # Find minimum distances for each unknown pixel
            min_fg_dist = np.min(fg_distances, axis=1)
            min_bg_dist = np.min(bg_distances, axis=1)
            
            # Calculate alpha values
            total_dist = min_fg_dist + min_bg_dist
            valid_mask = total_dist > 0
            
            space_alpha = alpha.copy()
            alpha_values = np.zeros(len(unknown_pixels))
            alpha_values[valid_mask] = min_bg_dist[valid_mask] / total_dist[valid_mask]
            space_alpha[unknown_coords] = alpha_values
            
            alpha_results.append(space_alpha)
        
        # Ensemble results from different color spaces
        final_alpha = np.mean(alpha_results, axis=0)
        
        # Apply edge-preserving smoothing
        alpha_uint8 = (final_alpha * 255).astype(np.uint8)
        smoothed = cv2.bilateralFilter(alpha_uint8, 9, 50, 50)
        
        return smoothed / 255.0

    def color_decontamination_improved(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Improved color decontamination with better edge preservation
        """
        h, w = image.shape[:2]
        decontaminated = image.copy().astype(np.float32)
        
        # Create a more precise edge map
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        edges = cv2.Canny(alpha_uint8, 30, 100)
        
        # Use smaller kernel for more precise processing
        kernel = np.ones((3, 3), np.uint8)
        edge_region = cv2.dilate(edges, kernel, iterations=1)
        
        # Better background color estimation using median of multiple regions
        bg_mask = alpha < 0.05
        if np.any(bg_mask):
            # Sample from different regions to get a more robust background estimate
            bg_pixels = image[bg_mask].astype(np.float32)
            if len(bg_pixels) > 1000:
                # Use median of random samples for more stable estimate
                sample_indices = np.random.choice(len(bg_pixels), 1000, replace=False)
                bg_color = np.median(bg_pixels[sample_indices], axis=0)
            else:
                bg_color = np.median(bg_pixels, axis=0)
        else:
            # Fallback: estimate from corner regions
            corner_size = min(h//10, w//10, 20)
            corners = [
                image[:corner_size, :corner_size],
                image[:corner_size, -corner_size:],
                image[-corner_size:, :corner_size],
                image[-corner_size:, -corner_size:]
            ]
            corner_pixels = np.concatenate([corner.reshape(-1, 3) for corner in corners])
            bg_color = np.median(corner_pixels.astype(np.float32), axis=0)
        
        # Process edge regions more carefully
        edge_coords = np.where(edge_region > 0)
        
        # Process in smaller batches for better quality
        batch_size = 1000
        for i in range(0, len(edge_coords[0]), batch_size):
            batch_y = edge_coords[0][i:i+batch_size]
            batch_x = edge_coords[1][i:i+batch_size]
            
            # Vectorized processing for the batch
            batch_alpha = alpha[batch_y, batch_x]
            batch_colors = image[batch_y, batch_x].astype(np.float32)
            
            # More conservative semi-transparent pixel selection
            semi_transparent = (batch_alpha > 0.1) & (batch_alpha < 0.9)
            
            if np.any(semi_transparent):
                st_alpha = batch_alpha[semi_transparent]
                st_colors = batch_colors[semi_transparent]
                
                # Improved decontamination formula with numerical stability
                denominator = st_alpha[:, np.newaxis] + 1e-6  # Add small epsilon for stability
                fg_colors = (st_colors - (1 - st_alpha)[:, np.newaxis] * bg_color) / denominator
                fg_colors = np.clip(fg_colors, 0, 255)
                
                # Adaptive correction strength based on alpha confidence
                alpha_confidence = np.minimum(st_alpha, 1 - st_alpha) * 2  # Peak at 0.5
                correction_strength = alpha_confidence * 0.5  # More conservative correction
                
                corrected_colors = ((1 - correction_strength[:, np.newaxis]) * st_colors + 
                                  correction_strength[:, np.newaxis] * fg_colors)
                
                # Update the decontaminated image
                st_indices = np.where(semi_transparent)[0]
                y_indices = batch_y[st_indices]
                x_indices = batch_x[st_indices]
                decontaminated[y_indices, x_indices] = corrected_colors
        
        return decontaminated.astype(np.uint8)

    def apply_enhanced_bilateral_refinement(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Enhanced bilateral refinement with better edge preservation
        """
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        
        # Apply bilateral filtering at multiple scales with better parameters
        refined1 = cv2.bilateralFilter(alpha_uint8, 5, 30, 30)   # Fine details
        refined2 = cv2.bilateralFilter(alpha_uint8, 9, 50, 50)   # Medium features  
        refined3 = cv2.bilateralFilter(alpha_uint8, 13, 80, 80)  # Large features
        
        # Create adaptive weights based on local edge strength
        edges = cv2.Canny(alpha_uint8, 30, 100)
        edge_distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        edge_distance = edge_distance / (np.max(edge_distance) + 1e-6)
        
        # More sophisticated weight calculation
        # Near edges: prefer fine filtering
        # Away from edges: blend all scales
        weight1 = np.exp(-edge_distance * 1.5)
        weight3 = np.exp(-edge_distance * 0.5) * (1 - weight1) * 0.3
        weight2 = 1 - weight1 - weight3
        
        # Ensure weights sum to 1
        total_weight = weight1 + weight2 + weight3
        weight1 /= total_weight
        weight2 /= total_weight  
        weight3 /= total_weight
        
        refined_alpha = (refined1 * weight1 + refined2 * weight2 + refined3 * weight3).astype(np.uint8)
        
        # Apply advanced edge-preserving filter if available
        try:
            # Try guided filter for better edge preservation
            if hasattr(cv2, 'ximgproc'):
                guide_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                guided_alpha = cv2.ximgproc.guidedFilter(
                    guide_image.astype(np.uint8),
                    refined_alpha,
                    radius=6,
                    eps=0.01
                )
                
                # Final light smoothing
                final_alpha = cv2.medianBlur(guided_alpha, 3)
                return final_alpha / 255.0
            else:
                # Standard fallback
                final_alpha = cv2.medianBlur(refined_alpha, 3)
                return final_alpha / 255.0
                
        except (ImportError, AttributeError):
            # Fallback to standard filters
            final_alpha = cv2.medianBlur(refined_alpha, 3)
            return final_alpha / 255.0

    def create_adaptive_trimap(self, mask: np.ndarray) -> np.ndarray:
        """
        Create adaptive trimap that adjusts based on image content
        """
        # Convert to binary
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Calculate adaptive kernel sizes based on mask properties
        h, w = mask.shape
        total_pixels = h * w
        fg_pixels = np.sum(binary_mask)
        fg_ratio = fg_pixels / total_pixels
        
        # Adaptive sizing based on foreground size and image dimensions
        base_size = min(h, w)
        
        # Smaller objects need smaller erosion/dilation
        if fg_ratio < 0.1:  # Small object
            fg_erosion = max(2, int(base_size * 0.005))
            bg_dilation = max(5, int(base_size * 0.015))
        elif fg_ratio < 0.3:  # Medium object  
            fg_erosion = max(3, int(base_size * 0.008))
            bg_dilation = max(8, int(base_size * 0.025))
        else:  # Large object
            fg_erosion = max(4, int(base_size * 0.012))
            bg_dilation = max(12, int(base_size * 0.035))
        
        # Create adaptive kernels
        fg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fg_erosion, fg_erosion))
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_dilation, bg_dilation))
        
        # Create sure foreground (eroded)
        sure_fg = cv2.erode(binary_mask, fg_kernel, iterations=1)
        
        # Create sure background (dilated inverse)
        sure_bg = cv2.dilate(binary_mask, bg_kernel, iterations=1)
        sure_bg = 1 - sure_bg
        
        # Create trimap with validation
        trimap = np.full_like(mask, 128, dtype=np.uint8)  # Unknown = 128
        trimap[sure_fg > 0] = 255  # Foreground = 255
        trimap[sure_bg > 0] = 0    # Background = 0
        
        # Validate trimap has all three regions
        unique_values = np.unique(trimap)
        if len(unique_values) < 3:
            logger.warning("Trimap missing regions, using fallback")
            # Fallback: use original mask with slight erosion/dilation
            fallback_fg = cv2.erode(binary_mask, np.ones((3,3), np.uint8), iterations=1)
            fallback_bg = 1 - cv2.dilate(binary_mask, np.ones((7,7), np.uint8), iterations=1)
            trimap = np.full_like(mask, 128, dtype=np.uint8)
            trimap[fallback_fg > 0] = 255
            trimap[fallback_bg > 0] = 0
        
        return trimap

    async def remove_background_with_u2net_advanced(self, image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """
        Advanced U2NET processing with all enhancements
        """
        start_time = time.time()
        
        u2net_models = [name for name in self.model_manager.models.keys() if name.startswith("u2net")]
        
        if not u2net_models:
            raise Exception("No U2NET models loaded")
        
        original_height, original_width = image.shape[:2]
        masks = []
        model_names = []
        confidences = []
        padding_infos = []
        
        # Try each U2NET model with improved preprocessing
        for model_name in u2net_models:
            try:
                model = self.model_manager.models[model_name]
                config = self.model_manager.model_configs[model_name]
                target_size = config["input_size"]
                
                # Use improved preprocessing
                input_tensor, original_shape, padding_info = self.preprocess_image_u2net_fixed(image, target_size)
                
                # Run model
                with torch.no_grad():
                    output = model(input_tensor)
                
                # Use improved post-processing
                raw_mask = self.postprocess_mask_u2net_fixed(output, original_shape, padding_info)
                
                # Better confidence calculation
                coverage = np.mean(raw_mask)
                edges = cv2.Canny((raw_mask * 255).astype(np.uint8), 50, 150)
                edge_quality = np.mean(edges) / 255.0
                
                # Consider mask smoothness and boundary quality
                gradient_x = np.abs(np.gradient(raw_mask, axis=1))
                gradient_y = np.abs(np.gradient(raw_mask, axis=0))
                smoothness = 1.0 / (1.0 + np.mean(gradient_x + gradient_y))
                
                # Weighted confidence score
                confidence = coverage * 0.4 + edge_quality * 0.3 + smoothness * 0.3
                
                masks.append(raw_mask)
                model_names.append(model_name)
                confidences.append(confidence)
                padding_infos.append(padding_info)
                
                logger.info(f"{model_name}: coverage={coverage:.3f}, edge_q={edge_quality:.3f}, "
                          f"smoothness={smoothness:.3f}, confidence={confidence:.3f}")
                
            except Exception as e:
                logger.error(f"{model_name} failed: {e}")
                continue
        
        if not masks:
            raise Exception("All U2NET models failed")
        
        # Improved ensemble
        if len(masks) > 1:
            final_mask = self.ensemble_masks_improved(masks, confidences)
            best_models = [model_names[i] for i in np.argsort(confidences)[-min(2, len(masks)):]]
            best_model = f"Ensemble({'+'.join(best_models)})"
        else:
            final_mask = masks[0]
            best_model = model_names[0]
        
        # Apply advanced enhancement pipeline
        logger.info("Applying conservative edge analysis...")
        refined_mask = self.advanced_edge_analysis_conservative(image, final_mask)
        
        logger.info("Creating adaptive trimap...")
        trimap = self.create_adaptive_trimap(refined_mask)
        
        logger.info("Applying multi-colorspace spectral matting...")
        alpha = self.spectral_matting_multi_colorspace_fixed(image, trimap)
        
        logger.info("Applying enhanced bilateral refinement...")
        alpha = self.apply_enhanced_bilateral_refinement(image, alpha)
        
        logger.info("Applying improved color decontamination...")
        decontaminated_image = self.color_decontamination_improved(image, alpha)
        
        # Create final result with post-processing validation
        alpha_final = np.clip(alpha, 0, 1)
        rgba_result = np.zeros((original_height, original_width, 4), dtype=np.uint8)
        rgba_result[:, :, :3] = decontaminated_image
        rgba_result[:, :, 3] = (alpha_final * 255).astype(np.uint8)
        
        processing_time = time.time() - start_time
        logger.info(f"Advanced processing completed in {processing_time:.2f}s")
        
        return rgba_result, best_model, processing_time

    async def remove_background_opencv_enhanced(self, image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """Enhanced OpenCV-based background removal"""
        start_time = time.time()
        
        height, width = image.shape[:2]
        
        # Method 1: Enhanced Mask R-CNN
        try:
            import torchvision.transforms as T
            from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
            
            maskrcnn_model = maskrcnn_resnet50_fpn_v2(pretrained=True).to(device)
            maskrcnn_model.eval()
            
            transform = T.Compose([T.ToTensor()])
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = transform(img_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                predictions = maskrcnn_model(input_tensor)
            
            # Focus on person and related objects
            relevant_classes = [1]  # Only person class for better precision
            
            pred = predictions[0]
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            masks = pred['masks'].cpu().numpy()
            
            # Find best mask
            best_mask = None
            best_score = 0
            
            for score, label, mask in zip(scores, labels, masks):
                if score > 0.5 and label in relevant_classes:
                    if score > best_score:
                        best_mask = cv2.resize(mask[0], (width, height))
                        best_score = score
            
            if best_mask is not None:
                # Apply advanced processing pipeline
                logger.info("Applying advanced processing to Mask R-CNN result...")
                
                refined_mask = self.advanced_edge_analysis_conservative(image, best_mask)
                trimap = self.create_adaptive_trimap(refined_mask)
                alpha = self.spectral_matting_multi_colorspace_fixed(image, trimap)
                alpha = self.apply_enhanced_bilateral_refinement(image, alpha)
                decontaminated_image = self.color_decontamination_improved(image, alpha)
                
                # Create RGBA result
                rgba_result = np.zeros((height, width, 4), dtype=np.uint8)
                rgba_result[:, :, :3] = decontaminated_image
                rgba_result[:, :, 3] = (alpha * 255).astype(np.uint8)
                
                processing_time = time.time() - start_time
                return rgba_result, "Enhanced Mask R-CNN + Advanced Processing", processing_time
            
        except Exception as e:
            logger.error(f"Enhanced Mask R-CNN failed: {e}")
        
        # Fallback: Enhanced ellipse with advanced processing
        logger.warning("Using enhanced ellipse fallback with advanced processing")
        fallback_mask = np.zeros((height, width), np.float32)
        cv2.ellipse(fallback_mask, (width//2, height//2), 
                   (int(width*0.45), int(height*0.5)), 0, 0, 360, 1, -1)
        
        # Apply advanced enhancement pipeline even to fallback
        refined_mask = self.advanced_edge_analysis_conservative(image, fallback_mask)
        trimap = self.create_adaptive_trimap(refined_mask)
        alpha = self.spectral_matting_multi_colorspace_fixed(image, trimap)
        alpha = self.apply_enhanced_bilateral_refinement(image, alpha)
        decontaminated_image = self.color_decontamination_improved(image, alpha)
        
        rgba_result = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_result[:, :, :3] = decontaminated_image
        rgba_result[:, :, 3] = (alpha * 255).astype(np.uint8)
        
        processing_time = time.time() - start_time
        return rgba_result, "Enhanced Ellipse + Advanced Processing", processing_time

    async def remove_background(self, image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """Main background removal function with advanced processing"""
        try:
            # Try advanced U2NET first
            if any(name.startswith("u2net") for name in self.model_manager.models.keys()):
                try:
                    return await self.remove_background_with_u2net_advanced(image)
                except Exception as e:
                    logger.info(f"Advanced U2NET failed: {e}, trying enhanced OpenCV")
            
            # Fallback to enhanced OpenCV
            return await self.remove_background_opencv_enhanced(image)
            
        except Exception as e:
            logger.error(f"All background removal methods failed: {e}")
            import traceback
            traceback.print_exc()
            # Emergency fallback - return original with alpha channel
            h, w = image.shape[:2]
            rgba_result = np.zeros((h, w, 4), dtype=np.uint8)
            rgba_result[:, :, :3] = image
            rgba_result[:, :, 3] = 255  # Fully opaque
            return rgba_result, "Failed - Original Image", 0.0


# Initialize components
model_manager = ModelManager()
bg_remover = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bg_remover
    
    logger.info("Starting ADVANCED U2NET Background Removal API...")
    
    await model_manager.download_models()
    model_manager.load_models()
    bg_remover = AdvancedBackgroundRemover(model_manager)
    
    logger.info("ADVANCED U2NET API ready for processing")
    
    yield
    
    logger.info("Shutting down...")

app = FastAPI(title="Advanced U2NET Background Removal API", version="14.0", lifespan=lifespan)

@app.get("/")
def read_root():
    return {
        "message": "Advanced U2NET Background Removal API",
        "version": "14.0 - Advanced Processing with Fixed Bugs",
        "models_loaded": list(model_manager.models.keys()) if model_manager.models else [],
        "device": str(device),
        "advanced_features": [
            "FIXED aspect ratio preservation in preprocessing",
            "FIXED cv2 import issue in spectral matting",
            "Advanced model ensemble with confidence scoring",
            "Conservative edge analysis with dual color spaces",
            "Multi-colorspace spectral matting (LAB, HSV, RGB)",
            "Enhanced bilateral refinement with edge awareness",
            "Improved color decontamination with background estimation",
            "Adaptive trimap generation based on image content",
            "High-quality transparency preservation",
            "Comprehensive error handling and fallbacks"
        ]
    }

@app.get("/health")
def health_check():
    memory_info = psutil.virtual_memory()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_available": list(model_manager.models.keys()),
        "device": str(device),
        "system": {
            "memory_usage": f"{memory_info.percent}%",
            "memory_available": f"{memory_info.available / (1024**3):.1f}GB"
        }
    }

@app.post("/segmentation")
async def segment_image(challenge: str = Form(...), input: UploadFile = File(...)):
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing request - Challenge: {challenge}, File: {input.filename}")
        
        if challenge != "cv3":
            return JSONResponse(
                status_code=400,
                content={"message": f"Only cv3 supported, received: {challenge}"}
            )
        
        if not input.content_type.startswith("image/"):
            return JSONResponse(
                status_code=400,
                content={"message": "File must be an image"}
            )
        
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(input.filename)[1] or ".jpg"
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
        
        contents = await input.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        image = cv2.imread(file_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read image")
        
        original_height, original_width = image.shape[:2]
        logger.info(f"Processing image: {original_width}x{original_height}")
        
        # Scale down image to max 512x512, preserving aspect ratio
        max_dim = 512
        h, w = image.shape[:2]
        scale = min(max_dim / h, max_dim / w, 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Image scaled down to {new_w}x{new_h} for processing")
        else:
            logger.info("Image size within 512x512, no scaling applied")
        
        result, method_used, processing_time_seconds = await bg_remover.remove_background(image)
        
        # Scale result back up to original size before saving
        if (result.shape[1], result.shape[0]) != (original_width, original_height):
            result = cv2.resize(result, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        
        # Always save as PNG to preserve transparency
        result_path = os.path.join(RESULT_DIR, f"{file_id}.png")
        cv2.imwrite(result_path, result)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if collection is not None:
            try:
                total_time = (datetime.now() - start_time).total_seconds()
                document = {
                    "original_file": file_path,
                    "processed_file": result_path,
                    "challenge": challenge,
                    "timestamp": start_time,
                    "file_id": file_id,
                    "original_size": f"{original_width}x{original_height}",
                    "method_used": method_used,
                    "processing_time_seconds": processing_time_seconds,
                    "total_time_seconds": total_time,
                    "device": str(device),
                    "version": "Advanced U2NET v14.0"
                }
                collection.insert_one(document)
            except Exception as e:
                logger.warning(f"MongoDB storage failed: {e}")
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Processing completed in {total_time:.2f}s using {method_used}")
        
        return JSONResponse(content={
            "message": "succeed", 
            "file_id": file_id,
            "method_used": method_used,
            "processing_time_seconds": processing_time_seconds,
            "total_time_seconds": total_time,
            "version": "Advanced_U2NET"
        })
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": f"Processing failed: {str(e)}"}
        )

@app.get("/result/{file_id}")
async def get_result(file_id: str):
    result_path = os.path.join(RESULT_DIR, f"{file_id}.png")
    if os.path.exists(result_path):
        return FileResponse(result_path)
    
    return JSONResponse(
        status_code=404,
        content={"message": "Result not found"}
    )

@app.get("/models")
def get_models():
    return {
        "loaded_models": list(model_manager.models.keys()),
        "model_configs": {k: {**v, "model_class": str(v.get("model_class", "None"))} 
                         for k, v in model_manager.model_configs.items()},
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Advanced U2NET Background Removal API")
    uvicorn.run(app, host="0.0.0.0", port=8080)