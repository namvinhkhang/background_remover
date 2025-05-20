import os
import io
import gc
import time
import uuid
import logging
import asyncio
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

app = FastAPI(title="Hybrid U2NET Background Removal API", version="16.0", lifespan=lifespan)

@app.get("/")
def read_root():
    return {
        "message": "Hybrid U2NET Background Removal API",
        "version": "16.0 - Hybrid Processing with Global Context",
        "models_loaded": list(model_manager.models.keys()) if model_manager.models else [],
        "device": str(device),
        "features": [
            "Scale-down processing for global context",
            "Preserves full image context for better object detection",
            "Optional guided refinement for high-resolution details",
            "No artifacts from tile boundaries",
            "Adaptive processing based on image size",
            "Edge-preserving smoothing and guided filtering",
            "Multiple processing quality levels"
        ],
        "processing_modes": {
            mode: config["description"] for mode, config in PROCESSING_CONFIGS.items()
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
        
        if processing_mode in PROCESSING_CONFIGS:
            global bg_remover
            bg_remover = create_optimized_bg_remover(model_manager, processing_mode)
        
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
        total_pixels = original_height * original_width
        logger.info(f"Processing image: {original_width}x{original_height} ({total_pixels / (1024*1024):.1f}MP)")
        
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
                    "challenge": challenge,
                    "timestamp": start_time,
                    "file_id": file_id,
                    "original_size": f"{original_width}x{original_height}",
                    "megapixels": total_pixels / (1024 * 1024),
                    "method_used": method_used,
                    "processing_mode": processing_mode,
                    "processing_time_seconds": processing_time_seconds,
                    "total_time_seconds": total_time,
                    "device": str(device),
                    "version": "Hybrid U2NET v16.0",
                    "used_refinement": "Refinement" in method_used
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
            "processing_mode": processing_mode,
            "processing_time_seconds": processing_time_seconds,
            "total_time_seconds": total_time,
            "version": "Hybrid_U2NET",
            "image_info": {
                "size": f"{original_width}x{original_height}",
                "megapixels": f"{total_pixels / (1024 * 1024):.1f}MP",
                "used_refinement": "Refinement" in method_used
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
        "device": str(device),
        "processing_configs": PROCESSING_CONFIGS
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
        bg_remover.max_processing_size = max_processing_size
        bg_remover.enable_refinement = enable_refinement
        bg_remover.refinement_threshold = int(refinement_threshold_mp * 1024 * 1024)
        
        logger.info(f"Processing reconfigured: max_size={max_processing_size}, "
                   f"refinement={enable_refinement}, threshold={refinement_threshold_mp}MP")
        
        return JSONResponse(content={
            "message": "Processing configuration updated successfully",
            "config": {
                "max_processing_size": max_processing_size,
                "enable_refinement": enable_refinement,
                "refinement_threshold_mp": refinement_threshold_mp
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to reconfigure: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Failed to update configuration: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Hybrid U2NET Background Removal API")
    uvicorn.run(app, host="0.0.0.0", port=8080)