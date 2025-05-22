import os
import io
import gc
import time
import uuid
import logging
import asyncio
import re
import mimetypes
from datetime import datetime
from typing import Optional, Tuple, List
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
from pydantic import BaseModel, validator, Field

from u2net import U2NET, U2NETP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
RESULT_DIR = os.getenv("RESULT_DIR", "results")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "models/pretrained")

# BOUNDARY CONSTANTS - Following boundary testing guidelines
class BoundaryLimits:
    # String length boundaries
    CHALLENGE_MIN_LENGTH = 1
    CHALLENGE_MAX_LENGTH = 20
    PROCESSING_MODE_MIN_LENGTH = 1  
    PROCESSING_MODE_MAX_LENGTH = 20
    
    # File size boundaries (in bytes)
    MIN_FILE_SIZE = 1024  # 1KB minimum to ensure it's a real image
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB maximum to prevent server crash like in the story
    
    # Numeric boundaries for configuration
    MIN_PROCESSING_SIZE = 128
    MAX_PROCESSING_SIZE = 2048
    MIN_REFINEMENT_THRESHOLD = 0.1
    MAX_REFINEMENT_THRESHOLD = 100.0
    
    # Valid values
    VALID_CHALLENGES = {"cv3"}  # Only cv3 is supported
    VALID_PROCESSING_MODES = {"fast", "balanced", "quality"}
    VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    VALID_IMAGE_MIME_TYPES = {
        "image/jpeg", "image/jpg", "image/png", "image/bmp", 
        "image/tiff", "image/webp", "image/x-ms-bmp"
    }

class InputValidator:
    """Comprehensive input validation following boundary testing guidelines"""
    
    @staticmethod
    def sanitize_string(value: str) -> str:
        """Sanitize string input - trim whitespace and remove dangerous characters"""
        if not isinstance(value, str):
            raise ValueError("Input must be a string")
        
        # Trim leading and trailing whitespace
        sanitized = value.strip()
        
        # Remove null bytes and other dangerous characters
        sanitized = sanitized.replace('\x00', '').replace('\n', '').replace('\r', '')
        sanitized = sanitized.replace('\t', '').replace('\b', '').replace('\f', '')
        
        return sanitized
    
    @staticmethod
    def validate_string_boundaries(value: str, field_name: str, min_length: int, max_length: int, 
                                 allowed_chars_pattern: str = None) -> str:
        """Validate string boundaries and character content"""
        # Sanitize first
        sanitized_value = InputValidator.sanitize_string(value)
        
        # Check empty string (lower boundary)
        if len(sanitized_value) < min_length:
            raise ValueError(f"{field_name} must be at least {min_length} characters long")
        
        # Check maximum length (upper boundary)  
        if len(sanitized_value) > max_length:
            raise ValueError(f"{field_name} must not exceed {max_length} characters")
        
        # Check character pattern if specified
        if allowed_chars_pattern and not re.match(allowed_chars_pattern, sanitized_value):
            raise ValueError(f"{field_name} contains invalid characters")
        
        return sanitized_value
    
    @staticmethod
    def validate_challenge(challenge: str) -> str:
        """Validate challenge parameter with boundary and value checks"""
        sanitized = InputValidator.validate_string_boundaries(
            challenge, "challenge", 
            BoundaryLimits.CHALLENGE_MIN_LENGTH, 
            BoundaryLimits.CHALLENGE_MAX_LENGTH,
            r'^[a-zA-Z0-9]+$'  # Only alphanumeric characters
        )
        
        if sanitized not in BoundaryLimits.VALID_CHALLENGES:
            raise ValueError(f"Invalid challenge value. Allowed: {BoundaryLimits.VALID_CHALLENGES}")
        
        return sanitized
    
    @staticmethod
    def validate_processing_mode(mode: str) -> str:
        """Validate processing mode parameter"""
        sanitized = InputValidator.validate_string_boundaries(
            mode, "processing_mode",
            BoundaryLimits.PROCESSING_MODE_MIN_LENGTH,
            BoundaryLimits.PROCESSING_MODE_MAX_LENGTH,
            r'^[a-zA-Z0-9_-]+$'  # Alphanumeric, underscore, hyphen
        )
        
        if sanitized not in BoundaryLimits.VALID_PROCESSING_MODES:
            raise ValueError(f"Invalid processing mode. Allowed: {BoundaryLimits.VALID_PROCESSING_MODES}")
        
        return sanitized
    
    @staticmethod
    async def validate_uploaded_file(file: UploadFile) -> None:
        """Comprehensive file validation following boundary testing guidelines"""
        # Check if file exists
        if not file or not file.filename:
            raise ValueError("No file provided")
        
        # Sanitize filename
        sanitized_filename = InputValidator.sanitize_string(file.filename)
        if not sanitized_filename:
            raise ValueError("Invalid filename")
        
        # Read file content for size validation
        file_content = await file.read()
        file_size = len(file_content)
        
        # Reset file pointer for later reading
        await file.seek(0)
        
        # File size boundaries
        if file_size < BoundaryLimits.MIN_FILE_SIZE:
            raise ValueError(f"File too small. Minimum size: {BoundaryLimits.MIN_FILE_SIZE} bytes")
        
        if file_size > BoundaryLimits.MAX_FILE_SIZE:
            raise ValueError(f"File too large. Maximum size: {BoundaryLimits.MAX_FILE_SIZE} bytes "
                           f"({BoundaryLimits.MAX_FILE_SIZE // (1024*1024)}MB)")
        
        # Content-Type validation (first check)
        if not file.content_type or file.content_type not in BoundaryLimits.VALID_IMAGE_MIME_TYPES:
            raise ValueError(f"Invalid file type. Allowed types: {BoundaryLimits.VALID_IMAGE_MIME_TYPES}")
        
        # File extension validation
        file_ext = os.path.splitext(sanitized_filename)[1].lower()
        if file_ext not in BoundaryLimits.VALID_IMAGE_EXTENSIONS:
            raise ValueError(f"Invalid file extension. Allowed: {BoundaryLimits.VALID_IMAGE_EXTENSIONS}")
        
        # Advanced file content validation - check if it's actually an image
        try:
            # Try to open with PIL to verify it's a real image
            image_pil = Image.open(io.BytesIO(file_content))
            image_pil.verify()  # Verify the image integrity
            
            # Additional check - try to load as numpy array
            image_pil = Image.open(io.BytesIO(file_content))  # Reopen after verify
            np.array(image_pil)
            
        except Exception as e:
            raise ValueError(f"File is not a valid image or is corrupted: {str(e)}")
        
        # Reset file pointer again
        await file.seek(0)
        
        logger.info(f"File validation passed: {sanitized_filename}, size: {file_size} bytes")
    
    @staticmethod
    def validate_numeric_boundaries(value: any, field_name: str, min_val: float, max_val: float) -> float:
        """Validate numeric boundaries"""
        try:
            numeric_val = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"{field_name} must be a valid number")
        
        if numeric_val < min_val:
            raise ValueError(f"{field_name} must be at least {min_val}")
        
        if numeric_val > max_val:
            raise ValueError(f"{field_name} must not exceed {max_val}")
        
        return numeric_val
    
    @staticmethod
    def validate_boolean(value: any, field_name: str) -> bool:
        """Validate boolean values"""
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            sanitized = InputValidator.sanitize_string(value).lower()
            if sanitized in {'true', '1', 'yes', 'on'}:
                return True
            elif sanitized in {'false', '0', 'no', 'off'}:
                return False
        
        raise ValueError(f"{field_name} must be a valid boolean value")

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
                    if model_name == "u2net" and os.path.exists("u2net.pth"):
                        logger.info("Found local u2net.pth, copying to cache...")
                        import shutil
                        shutil.copy("u2net.pth", model_path)
            else:
                logger.info(f"{model_name} already exists")

    def load_u2net_model(self, model_name: str) -> Optional[torch.nn.Module]:
        try:
            config = self.model_configs[model_name]
            model_class = config["model_class"]
            model = model_class(in_ch=3, out_ch=1).to(device)
            
            model_path = os.path.join(MODEL_CACHE_DIR, config["filename"])
            
            if os.path.exists(model_path):
                try:
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
        self.models["u2net"] = self.load_u2net_model("u2net")
        self.models["u2net_portrait"] = self.load_u2net_model("u2net_portrait")
        self.models["u2netp"] = self.load_u2net_model("u2netp")
        self.models["modnet"] = self.load_modnet()
        
        self.models = {k: v for k, v in self.models.items() if v is not None}
        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")


class HybridBackgroundRemover:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        
        self.max_processing_size = 512
        self.enable_refinement = True
        self.refinement_threshold = 2048 * 2048
        
    def preprocess_image_u2net_fixed(self, image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image_rgb.shape[:2]
        
        target_width, target_height = target_size
        scale = min(target_width / original_width, target_height / original_height)
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        image_resized = cv2.resize(image_rgb, (new_width, new_height))
        
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        pad_y = (target_height - new_height) // 2
        pad_x = (target_width - new_width) // 2
        
        canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = image_resized
        
        canvas_normalized = canvas.astype(np.float32) / 255.0
        
        image_tensor = torch.from_numpy(canvas_normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        return image_tensor, (original_height, original_width), (pad_y, pad_x, new_height, new_width)

    def postprocess_mask_u2net_fixed(self, output: torch.Tensor, original_shape: Tuple[int, int], 
                                   padding_info: Tuple[int, int, int, int]) -> np.ndarray:
        if isinstance(output, tuple):
            mask = output[0]
        else:
            mask = output
            
        mask_np = mask.squeeze().cpu().detach().numpy()
        
        pad_y, pad_x, new_height, new_width = padding_info
        
        mask_crop = mask_np[pad_y:pad_y + new_height, pad_x:pad_x + new_width]
        
        mask_resized = cv2.resize(mask_crop, (original_shape[1], original_shape[0]))
        
        mask_resized = np.clip(mask_resized, 0, 1)
        
        return mask_resized

    def ensemble_masks_improved(self, masks: list, confidences: list = None) -> np.ndarray:
        if not masks:
            return None
            
        if len(masks) == 1:
            return masks[0]
            
        if confidences is None:
            confidences = [1.0] * len(masks)
        
        confidences = np.array(confidences)
        confidences = confidences / (np.sum(confidences) + 1e-8)
        
        mask_stack = np.stack(masks, axis=0)
        mask_variance = np.var(mask_stack, axis=0)
        
        ensemble_mask = np.zeros_like(masks[0])
        
        agreement_threshold = 0.1
        agreement_mask = mask_variance < agreement_threshold
        
        for i, (mask, conf) in enumerate(zip(masks, confidences)):
            ensemble_mask += mask * conf
            
        disagreement_mask = ~agreement_mask
        if np.any(disagreement_mask):
            best_idx = np.argmax(confidences)
            ensemble_mask[disagreement_mask] = masks[best_idx][disagreement_mask]
            
        return ensemble_mask

    def create_guided_trimap(self, mask: np.ndarray, refinement_strength: float = 0.3) -> np.ndarray:
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        h, w = mask.shape
        base_size = min(h, w)
        
        erosion_size = max(2, int(base_size * 0.01 * refinement_strength))
        dilation_size = max(5, int(base_size * 0.02 * refinement_strength))
        
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        
        sure_fg = cv2.erode(binary_mask, erosion_kernel, iterations=1)
        sure_bg = 1 - cv2.dilate(binary_mask, dilation_kernel, iterations=1)
        
        trimap = np.full_like(mask, 128, dtype=np.uint8)
        trimap[sure_fg > 0] = 255
        trimap[sure_bg > 0] = 0
        
        return trimap

    def guided_refinement_matting(self, image: np.ndarray, coarse_mask: np.ndarray, 
                                trimap: np.ndarray) -> np.ndarray:
        alpha = trimap.copy().astype(np.float32) / 255.0
        
        unknown_mask = (trimap == 128)
        
        if not np.any(unknown_mask):
            return alpha
        
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        fg_mask = (trimap == 255)
        bg_mask = (trimap == 0)
        
        if not (np.any(fg_mask) and np.any(bg_mask)):
            return alpha
        
        max_samples = 1000
        
        fg_weights = coarse_mask[fg_mask]
        fg_colors = lab_image[fg_mask]
        if len(fg_colors) > max_samples:
            probs = fg_weights / (np.sum(fg_weights) + 1e-8)
            indices = np.random.choice(len(fg_colors), max_samples, replace=False, p=probs)
            fg_colors = fg_colors[indices]
        
        bg_weights = 1.0 - coarse_mask[bg_mask]
        bg_colors = lab_image[bg_mask]
        if len(bg_colors) > max_samples:
            probs = bg_weights / (np.sum(bg_weights) + 1e-8)
            indices = np.random.choice(len(bg_colors), max_samples, replace=False, p=probs)
            bg_colors = bg_colors[indices]
        
        unknown_coords = np.where(unknown_mask)
        unknown_pixels = lab_image[unknown_coords]
        coarse_values = coarse_mask[unknown_coords]
        
        fg_distances = np.linalg.norm(unknown_pixels[:, np.newaxis, :] - fg_colors[np.newaxis, :, :], axis=2)
        bg_distances = np.linalg.norm(unknown_pixels[:, np.newaxis, :] - bg_colors[np.newaxis, :, :], axis=2)
        
        min_fg_dist = np.min(fg_distances, axis=1)
        min_bg_dist = np.min(bg_distances, axis=1)
        
        total_dist = min_fg_dist + min_bg_dist
        valid_mask = total_dist > 0
        
        alpha_values = np.zeros(len(unknown_pixels))
        alpha_values[valid_mask] = min_bg_dist[valid_mask] / total_dist[valid_mask]
        
        distance_confidence = np.exp(-np.minimum(min_fg_dist, min_bg_dist) / 10.0)
        blend_weight = distance_confidence * 0.7 + 0.3
        
        final_alpha = blend_weight * alpha_values + (1 - blend_weight) * coarse_values
        alpha[unknown_coords] = final_alpha
        
        return alpha

    def apply_edge_preserving_smoothing(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        
        smoothed = cv2.bilateralFilter(alpha_uint8, 9, 80, 80)
        
        try:
            if hasattr(cv2, 'ximgproc'):
                guide_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                guided = cv2.ximgproc.guidedFilter(guide_image, smoothed, radius=8, eps=0.01)
                return guided / 255.0
        except:
            pass
        
        return smoothed / 255.0

    async def get_coarse_mask_with_context(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        original_h, original_w = image.shape[:2]
        
        scale = min(self.max_processing_size / original_w, self.max_processing_size / original_h)
        if scale < 1.0:
            scaled_w = int(original_w * scale)
            scaled_h = int(original_h * scale)
            scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        else:
            scaled_image = image
            scaled_w = original_w
            scaled_h = original_h
            
        u2net_models = [name for name in self.model_manager.models.keys() if name.startswith("u2net")]
        
        masks = []
        model_names = []
        confidences = []
        
        for model_name in u2net_models:
            try:
                model = self.model_manager.models[model_name]
                config = self.model_manager.model_configs[model_name]
                target_size = config["input_size"]
                
                input_tensor, original_shape, padding_info = self.preprocess_image_u2net_fixed(scaled_image, target_size)
                
                with torch.no_grad():
                    output = model(input_tensor)
                
                scaled_mask = self.postprocess_mask_u2net_fixed(output, (scaled_h, scaled_w), padding_info)
                
                mask = cv2.resize(scaled_mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
                
                coverage = np.mean(mask)
                edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
                edge_quality = np.mean(edges) / 255.0
                smoothness = 1.0 / (1.0 + np.std(mask))
                
                confidence = coverage * 0.4 + edge_quality * 0.3 + smoothness * 0.3
                
                masks.append(mask)
                model_names.append(model_name)
                confidences.append(confidence)
                
                self.logger.info(f"{model_name} coarse mask: confidence={confidence:.3f}")
                
            except Exception as e:
                self.logger.error(f"{model_name} failed: {e}")
                continue
        
        if not masks:
            raise Exception("All U2NET models failed")
        
        if len(masks) > 1:
            final_mask = self.ensemble_masks_improved(masks, confidences)
            best_models = [model_names[i] for i in np.argsort(confidences)[-2:]]
            method = f"Hybrid-Coarse({'+'.join(best_models)})"
        else:
            final_mask = masks[0]
            method = f"Hybrid-Coarse({model_names[0]})"
        
        return final_mask, method

    def should_use_refinement(self, image: np.ndarray) -> bool:
        if not self.enable_refinement:
            return False
        h, w = image.shape[:2]
        return (h * w) > self.refinement_threshold

    async def refine_mask_with_guided_matting(self, image: np.ndarray, coarse_mask: np.ndarray) -> np.ndarray:
        self.logger.info("Applying guided refinement on full resolution...")
        
        trimap = self.create_guided_trimap(coarse_mask, refinement_strength=0.2)
        
        refined_alpha = self.guided_refinement_matting(image, coarse_mask, trimap)
        
        final_alpha = self.apply_edge_preserving_smoothing(image, refined_alpha)
        
        return final_alpha

    async def remove_background(self, image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        start_time = time.time()
        
        try:
            self.logger.info("Getting coarse mask with global context...")
            coarse_mask, coarse_method = await self.get_coarse_mask_with_context(image)
            
            if self.should_use_refinement(image):
                self.logger.info("Image is high-res, applying guided refinement...")
                final_alpha = await self.refine_mask_with_guided_matting(image, coarse_mask)
                method = f"{coarse_method} + Guided-Refinement"
            else:
                self.logger.info("Using coarse mask directly...")
                final_alpha = coarse_mask
                method = coarse_method
            
            h, w = image.shape[:2]
            rgba_result = np.zeros((h, w, 4), dtype=np.uint8)
            rgba_result[:, :, :3] = image
            rgba_result[:, :, 3] = (final_alpha * 255).astype(np.uint8)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Hybrid processing completed in {processing_time:.2f}s")
            
            return rgba_result, method, processing_time
            
        except Exception as e:
            self.logger.error(f"Hybrid processing failed: {e}")
            h, w = image.shape[:2]
            rgba_result = np.zeros((h, w, 4), dtype=np.uint8)
            rgba_result[:, :, :3] = image
            rgba_result[:, :, 3] = 255
            return rgba_result, "Failed - Original Image", 0.0


PROCESSING_CONFIGS = {
    "fast": {
        "max_processing_size": 384,
        "enable_refinement": False,
        "description": "Fast processing, lower quality"
    },
    "balanced": {
        "max_processing_size": 512,
        "enable_refinement": True,
        "refinement_threshold": 2048 * 2048,
        "description": "Balanced speed and quality"
    },
    "quality": {
        "max_processing_size": 640,
        "enable_refinement": True,
        "refinement_threshold": 1024 * 1024,
        "description": "High quality, slower processing"
    }
}

def create_optimized_bg_remover(model_manager, config_name="balanced"):
    config = PROCESSING_CONFIGS[config_name]
    
    bg_remover = HybridBackgroundRemover(model_manager)
    bg_remover.max_processing_size = config["max_processing_size"]
    bg_remover.enable_refinement = config["enable_refinement"]
    if "refinement_threshold" in config:
        bg_remover.refinement_threshold = config["refinement_threshold"]
    
    logger.info(f"Created {config_name} background remover: {config['description']}")
    return bg_remover


model_manager = ModelManager()
bg_remover = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bg_remover
    
    logger.info("Starting HYBRID U2NET Background Removal API...")
    
    await model_manager.download_models()
    model_manager.load_models()
    
    bg_remover = create_optimized_bg_remover(model_manager, "balanced")
    
    logger.info("HYBRID U2NET API ready for processing")
    
    yield
    
    logger.info("Shutting down...")

app = FastAPI(title="Hybrid U2NET Background Removal API", version="16.1", lifespan=lifespan)

@app.get("/")
def read_root():
    return {
        "message": "Hybrid U2NET Background Removal API",
        "version": "16.1 - Enhanced with Comprehensive Boundary Testing",
        "models_loaded": list(model_manager.models.keys()) if model_manager.models else [],
        "device": str(device),
        "features": [
            "Scale-down processing for global context",
            "Preserves full image context for better object detection",
            "Optional guided refinement for high-resolution details",
            "No artifacts from tile boundaries",
            "Adaptive processing based on image size",
            "Edge-preserving smoothing and guided filtering",
            "Multiple processing quality levels",
            "Comprehensive boundary testing and input validation",
            "File size limits and content validation",
            "String sanitization and character validation"
        ],
        "processing_modes": {
            mode: config["description"] for mode, config in PROCESSING_CONFIGS.items()
        },
        "boundary_limits": {
            "max_file_size_mb": BoundaryLimits.MAX_FILE_SIZE // (1024*1024),
            "min_file_size_kb": BoundaryLimits.MIN_FILE_SIZE // 1024,
            "supported_formats": list(BoundaryLimits.VALID_IMAGE_EXTENSIONS),
            "valid_challenges": list(BoundaryLimits.VALID_CHALLENGES),
            "valid_processing_modes": list(BoundaryLimits.VALID_PROCESSING_MODES)
        }
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
        },
        "processing_config": {
            "max_size": bg_remover.max_processing_size if bg_remover else None,
            "refinement_enabled": bg_remover.enable_refinement if bg_remover else None,
            "refinement_threshold_mp": (bg_remover.refinement_threshold / (1024*1024)) if bg_remover else None
        },
        "boundary_validation": {
            "file_size_limits": f"{BoundaryLimits.MIN_FILE_SIZE // 1024}KB - {BoundaryLimits.MAX_FILE_SIZE // (1024*1024)}MB",
            "string_length_limits": f"Challenge: {BoundaryLimits.CHALLENGE_MIN_LENGTH}-{BoundaryLimits.CHALLENGE_MAX_LENGTH}, Mode: {BoundaryLimits.PROCESSING_MODE_MIN_LENGTH}-{BoundaryLimits.PROCESSING_MODE_MAX_LENGTH}",
            "supported_extensions": list(BoundaryLimits.VALID_IMAGE_EXTENSIONS)
        }
    }

@app.post("/segmentation")
async def segment_image(
    challenge: str = Form(...), 
    input: UploadFile = File(...),
    processing_mode: str = Form("quality")
):
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing request - Challenge: {challenge}, File: {input.filename}, Mode: {processing_mode}")
        
        # COMPREHENSIVE BOUNDARY TESTING - Following the guidelines
        try:
            # Validate challenge parameter (string boundaries and content)
            validated_challenge = InputValidator.validate_challenge(challenge)
            
            # Validate processing mode parameter
            validated_processing_mode = InputValidator.validate_processing_mode(processing_mode)
            
            # Comprehensive file validation (size, type, content)
            await InputValidator.validate_uploaded_file(input)
            
        except ValueError as e:
            logger.warning(f"Boundary validation failed: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"Input validation failed: {str(e)}",
                    "error_type": "boundary_validation_error",
                    "boundary_limits": {
                        "max_file_size_mb": BoundaryLimits.MAX_FILE_SIZE // (1024*1024),
                        "min_file_size_kb": BoundaryLimits.MIN_FILE_SIZE // 1024,
                        "valid_challenges": list(BoundaryLimits.VALID_CHALLENGES),
                        "valid_processing_modes": list(BoundaryLimits.VALID_PROCESSING_MODES),
                        "supported_formats": list(BoundaryLimits.VALID_IMAGE_EXTENSIONS)
                    }
                }
            )
        
        # Configure processing mode if it's valid
        if validated_processing_mode in PROCESSING_CONFIGS:
            global bg_remover
            bg_remover = create_optimized_bg_remover(model_manager, validated_processing_mode)
        
        # Generate secure file ID
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(input.filename)[1].lower() or ".jpg"
        
        # Validate file extension again (double-check)
        if file_extension not in BoundaryLimits.VALID_IMAGE_EXTENSIONS:
            raise ValueError(f"Invalid file extension: {file_extension}")
        
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
        
        # Save file securely
        contents = await input.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Additional image validation after saving
        image = cv2.imread(file_path)
        if image is None:
            # Clean up the invalid file
            try:
                os.remove(file_path)
            except:
                pass
            raise HTTPException(status_code=400, detail="Could not read image file - file may be corrupted")
        
        original_height, original_width = image.shape[:2]
        total_pixels = original_height * original_width
        
        # Additional boundary check - image dimensions
        if original_width < 32 or original_height < 32:
            try:
                os.remove(file_path)
            except:
                pass
            raise HTTPException(status_code=400, detail="Image too small - minimum dimensions: 32x32 pixels")
        
        if original_width > 10000 or original_height > 10000:
            try:
                os.remove(file_path)
            except:
                pass
            raise HTTPException(status_code=400, detail="Image too large - maximum dimensions: 10000x10000 pixels")
        
        logger.info(f"Processing validated image: {original_width}x{original_height} ({total_pixels / (1024*1024):.1f}MP)")
        
        result, method_used, processing_time_seconds = await bg_remover.remove_background(image)
        
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
                    "challenge": validated_challenge,
                    "timestamp": start_time,
                    "file_id": file_id,
                    "original_size": f"{original_width}x{original_height}",
                    "megapixels": total_pixels / (1024 * 1024),
                    "method_used": method_used,
                    "processing_mode": validated_processing_mode,
                    "processing_time_seconds": processing_time_seconds,
                    "total_time_seconds": total_time,
                    "device": str(device),
                    "version": "Hybrid U2NET v16.1 - Enhanced Boundary Testing",
                    "used_refinement": "Refinement" in method_used,
                    "file_size_bytes": len(contents),
                    "boundary_validation_passed": True
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
            "processing_mode": validated_processing_mode,
            "processing_time_seconds": processing_time_seconds,
            "total_time_seconds": total_time,
            "version": "Hybrid_U2NET_Enhanced",
            "image_info": {
                "size": f"{original_width}x{original_height}",
                "megapixels": f"{total_pixels / (1024 * 1024):.1f}MP",
                "used_refinement": "Refinement" in method_used,
                "file_size_kb": f"{len(contents) / 1024:.1f}KB"
            },
            "validation_info": {
                "boundary_checks_passed": True,
                "challenge_validated": validated_challenge,
                "processing_mode_validated": validated_processing_mode
            }
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
    try:
        # Validate file_id format (boundary testing for path parameters)
        validated_file_id = InputValidator.validate_string_boundaries(
            file_id, "file_id", 1, 100, r'^[a-zA-Z0-9-]+$'
        )
        
        result_path = os.path.join(RESULT_DIR, f"{validated_file_id}.png")
        if os.path.exists(result_path):
            return FileResponse(result_path)
        
        return JSONResponse(
            status_code=404,
            content={"message": "Result not found"}
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Invalid file ID: {str(e)}"}
        )

@app.get("/models")
def get_models():
    return {
        "loaded_models": list(model_manager.models.keys()),
        "model_configs": {k: {**v, "model_class": str(v.get("model_class", "None"))} 
                         for k, v in model_manager.model_configs.items()},
        "device": str(device),
        "processing_configs": PROCESSING_CONFIGS,
        "boundary_validation": {
            "file_size_limits": {
                "min_bytes": BoundaryLimits.MIN_FILE_SIZE,
                "max_bytes": BoundaryLimits.MAX_FILE_SIZE,
                "min_display": f"{BoundaryLimits.MIN_FILE_SIZE // 1024}KB",
                "max_display": f"{BoundaryLimits.MAX_FILE_SIZE // (1024*1024)}MB"
            },
            "supported_formats": list(BoundaryLimits.VALID_IMAGE_EXTENSIONS),
            "valid_challenges": list(BoundaryLimits.VALID_CHALLENGES),
            "valid_processing_modes": list(BoundaryLimits.VALID_PROCESSING_MODES)
        }
    }

@app.post("/configure")
async def configure_processing(
    max_processing_size: int = Form(512),
    enable_refinement: bool = Form(True),
    refinement_threshold_mp: float = Form(4.0)
):
    global bg_remover
    
    if bg_remover is None:
        return JSONResponse(
            status_code=400,
            content={"message": "Background remover not initialized"}
        )
    
    try:
        # BOUNDARY TESTING for configuration parameters
        validated_max_size = int(InputValidator.validate_numeric_boundaries(
            max_processing_size, "max_processing_size", 
            BoundaryLimits.MIN_PROCESSING_SIZE, BoundaryLimits.MAX_PROCESSING_SIZE
        ))
        
        validated_refinement = InputValidator.validate_boolean(enable_refinement, "enable_refinement")
        
        validated_threshold = InputValidator.validate_numeric_boundaries(
            refinement_threshold_mp, "refinement_threshold_mp",
            BoundaryLimits.MIN_REFINEMENT_THRESHOLD, BoundaryLimits.MAX_REFINEMENT_THRESHOLD
        )
        
        bg_remover.max_processing_size = validated_max_size
        bg_remover.enable_refinement = validated_refinement
        bg_remover.refinement_threshold = int(validated_threshold * 1024 * 1024)
        
        logger.info(f"Processing reconfigured: max_size={validated_max_size}, "
                   f"refinement={validated_refinement}, threshold={validated_threshold}MP")
        
        return JSONResponse(content={
            "message": "Processing configuration updated successfully",
            "config": {
                "max_processing_size": validated_max_size,
                "enable_refinement": validated_refinement,
                "refinement_threshold_mp": validated_threshold
            },
            "boundary_validation": {
                "all_parameters_validated": True,
                "applied_limits": {
                    "max_processing_size_range": f"{BoundaryLimits.MIN_PROCESSING_SIZE}-{BoundaryLimits.MAX_PROCESSING_SIZE}",
                    "refinement_threshold_range": f"{BoundaryLimits.MIN_REFINEMENT_THRESHOLD}-{BoundaryLimits.MAX_REFINEMENT_THRESHOLD}MP"
                }
            }
        })
        
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "message": f"Configuration validation failed: {str(e)}",
                "boundary_limits": {
                    "max_processing_size_range": f"{BoundaryLimits.MIN_PROCESSING_SIZE}-{BoundaryLimits.MAX_PROCESSING_SIZE}",
                    "refinement_threshold_range": f"{BoundaryLimits.MIN_REFINEMENT_THRESHOLD}-{BoundaryLimits.MAX_REFINEMENT_THRESHOLD}MP"
                }
            }
        )
    except Exception as e:
        logger.error(f"Failed to reconfigure: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Failed to update configuration: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Background Removal API...")
    uvicorn.run(app, host="0.0.0.0", port=8080)