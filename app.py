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
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import psutil
import gdown
from huggingface_hub import hf_hub_download
import albumentations as A
from scipy import ndimage
from skimage import segmentation, morphology
from scipy.spatial.distance import cdist

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Background Removal API", version="1.0")

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
    logger.info("MongoDB connected successfully")
except:
    logger.warning("MongoDB not available - continuing without database")
    client = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    logger.info(f"GPU optimizations enabled for {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    logger.info("CPU optimizations applied")

max_workers = 1 if device.type == "cuda" else 2
executor = ThreadPoolExecutor(max_workers=max_workers)


class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_configs = {
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
            else:
                logger.info(f"{model_name} already exists")

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
        self.models["modnet"] = self.load_modnet()
        self.models = {k: v for k, v in self.models.items() if v is not None}
        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
        
        if len(self.models) == 0:
            logger.warning("No deep learning models loaded - will use OpenCV fallback only")


class BackgroundRemover:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.preprocessing = A.Compose([
            A.LongestMaxSize(max_size=1024),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT),
        ])
        
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, target_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image_normalized).unsqueeze(0).to(device)
        return image_tensor

    def postprocess_mask(self, mask: torch.Tensor, original_shape: Tuple[int, int]) -> np.ndarray:
        mask_np = mask.squeeze().cpu().detach().numpy()
        mask_resized = cv2.resize(mask_np, (original_shape[1], original_shape[0]))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_refined = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_smooth = cv2.GaussianBlur(mask_refined, (5, 5), 0)
        
        return mask_smooth

    def create_trimap(self, mask: np.ndarray, erode_ksize: int = 10, dilate_ksize: int = 10) -> np.ndarray:
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
        
        eroded = cv2.erode(mask, kernel_erode, iterations=1)
        dilated = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        trimap = np.zeros_like(mask)
        trimap[eroded > 0.5] = 1
        trimap[dilated < 0.5] = 0
        trimap[(dilated >= 0.5) & (eroded <= 0.5)] = 0.5
        
        return trimap

    def apply_guided_filter(self, image: np.ndarray, mask: np.ndarray, radius: int = 8, eps: float = 0.2) -> np.ndarray:
        try:
            guided_mask = cv2.ximgproc.guidedFilter(
                guide=image.astype(np.uint8),
                src=(mask * 255).astype(np.uint8),
                radius=radius,
                eps=eps
            ) / 255.0
            return guided_mask
        except:
            logger.warning("Guided filter not available, using Gaussian blur fallback")
            return cv2.GaussianBlur(mask, (radius*2+1, radius*2+1), 0)

    def expand_mask_for_accessories(self, image: np.ndarray, mask: np.ndarray, max_expansion: int = 50) -> np.ndarray:
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        hull = cv2.convexHull(largest_contour)
        hull_mask = np.zeros_like(binary_mask)
        cv2.fillPoly(hull_mask, [hull], 1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_expansion, max_expansion))
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        
        masked_area = image[binary_mask > 0]
        if len(masked_area) > 0:
            mean_color = np.mean(masked_area, axis=0)
            std_color = np.std(masked_area, axis=0) + 1e-6
            
            color_diff = np.abs(image.astype(np.float32) - mean_color)
            color_distance = np.sqrt(np.sum((color_diff / std_color) ** 2, axis=2))
            color_mask = (color_distance < 2.0).astype(np.uint8)
            
            safe_expansion = np.logical_and(
                np.logical_and(dilated_mask, color_mask),
                hull_mask
            ).astype(np.uint8)
            
            final_mask = np.maximum(binary_mask, safe_expansion)
        else:
            final_mask = hull_mask
        
        return final_mask.astype(np.float32)

    def enhance_mask_edges(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        guided_mask = self.apply_guided_filter(image, mask, radius=8, eps=0.2)
        return guided_mask

    def create_professional_output(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        alpha = np.clip(alpha, 0, 1)
        
        height, width = image.shape[:2]
        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_image[:, :, :3] = image
        rgba_image[:, :, 3] = (alpha * 255).astype(np.uint8)
        
        return rgba_image

    async def remove_background_deep(self, image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        start_time = time.time()
        
        original_height, original_width = image.shape[:2]
        best_mask = None
        best_model = "none"
        best_confidence = 0.0
        
        if not self.model_manager.models:
            logger.info("No deep learning models available")
            raise Exception("No deep learning models loaded")
        
        for model_name, model in self.model_manager.models.items():
            try:
                logger.info(f"Trying {model_name}...")
                
                config = self.model_manager.model_configs[model_name]
                target_size = config["input_size"]
                
                input_tensor = self.preprocess_image(image, target_size)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    
                mask = self.postprocess_mask(output, (original_height, original_width))
                
                coverage = np.mean(mask)
                edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
                edge_quality = np.mean(edges) / 255.0
                confidence = coverage * 0.7 + edge_quality * 0.3
                
                logger.info(f"{model_name}: coverage={coverage:.3f}, confidence={confidence:.3f}")
                
                if confidence > 0.3 and coverage > 0.1:
                    if confidence > best_confidence:
                        best_mask = mask
                        best_model = model_name
                        best_confidence = confidence
                    
            except Exception as e:
                logger.error(f"{model_name} failed: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        if best_mask is None or best_confidence < 0.3:
            raise Exception(f"Deep learning models failed (best confidence: {best_confidence:.3f})")
            
        logger.info(f"Best model: {best_model} (confidence: {best_confidence:.3f})")
        return best_mask, best_model, processing_time

    async def remove_background_opencv(self, image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        start_time = time.time()
        
        logger.info("Using OpenCV methods...")
        
        height, width = image.shape[:2]
        best_mask = None
        best_method = "unknown"
        best_coverage = 0.0
        
        try:
            logger.info("Trying Mask R-CNN...")
            
            import torchvision.transforms as T
            from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
            
            maskrcnn_model = maskrcnn_resnet50_fpn_v2(pretrained=True).to(device)
            maskrcnn_model.eval()
            
            transform = T.Compose([T.ToTensor()])
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = transform(img_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                predictions = maskrcnn_model(input_tensor)
            
            relevant_classes = [1, 16, 17, 18, 19, 20, 21, 84, 85, 86, 87, 88, 89, 90]
            
            pred = predictions[0]
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            masks = pred['masks'].cpu().numpy()
            
            combined_mask = np.zeros((height, width), dtype=np.float32)
            
            for i, (score, label, mask) in enumerate(zip(scores, labels, masks)):
                if score > 0.3 and label in relevant_classes:
                    mask_resized = cv2.resize(mask[0], (width, height))
                    combined_mask = np.maximum(combined_mask, mask_resized)
                    logger.info(f"Found class {label} with confidence {score:.3f}")
            
            if np.max(combined_mask) > 0:
                rcnn_mask = (combined_mask > 0.3).astype(np.uint8)
                rcnn_mask = self.expand_mask_for_accessories(image, rcnn_mask, max_expansion=40)
                rcnn_alpha = self.enhance_mask_edges(image, rcnn_mask)
                
                coverage = np.mean(rcnn_alpha)
                logger.info(f"Mask R-CNN coverage: {coverage:.3f}")
                
                if coverage > best_coverage and coverage > 0.05:
                    best_mask = rcnn_alpha
                    best_method = "Mask R-CNN"
                    best_coverage = coverage
            
        except Exception as e:
            logger.error(f"Mask R-CNN failed: {e}")
        
        if best_coverage < 0.3:
            try:
                logger.info("Trying GrabCut...")
                
                margin_x = max(1, int(width * 0.002))
                margin_y = max(1, int(height * 0.002))
                rect = (margin_x, margin_y, width - 2*margin_x, height - 2*margin_y)
                
                mask = np.zeros((height, width), np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                
                cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
                result_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                
                expanded_mask = self.expand_mask_for_accessories(image, result_mask, max_expansion=50)
                enhanced_alpha = self.enhance_mask_edges(image, expanded_mask)
                
                coverage = np.mean(enhanced_alpha)
                logger.info(f"GrabCut coverage: {coverage:.3f}")
                
                if coverage > 0.1 and coverage < 0.8:
                    if coverage > best_coverage:
                        best_mask = enhanced_alpha
                        best_method = "GrabCut"
                        best_coverage = coverage
                
            except Exception as e:
                logger.error(f"GrabCut failed: {e}")
        
        if best_mask is None or best_coverage < 0.05:
            logger.warning("Using ellipse fallback")
            fallback_mask = np.zeros((height, width), np.uint8)
            cv2.ellipse(fallback_mask, (width//2, height//2), 
                       (int(width*0.47), int(height*0.47)), 0, 0, 360, 1, -1)
            
            best_mask = self.enhance_mask_edges(image, fallback_mask)
            best_method = "Ellipse fallback"
            best_coverage = np.mean(best_mask)
        
        processing_time = time.time() - start_time
        
        if best_mask is not None:
            result = self.create_professional_output(image, best_mask)
        else:
            result = image.copy()
        
        logger.info(f"Best method: {best_method}, coverage: {best_coverage:.3f}")
        
        return result, best_method, processing_time

    async def remove_background(self, image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        try:
            result, method_used, processing_time = await self.remove_background_opencv(image)
            
            if self.model_manager.models and "fallback" not in method_used.lower():
                try:
                    mask, dl_model, dl_time = await self.remove_background_deep(image)
                    enhanced_alpha = self.enhance_mask_edges(image, mask)
                    dl_result = self.create_professional_output(image, enhanced_alpha)
                    
                    dl_coverage = np.mean(enhanced_alpha)
                    
                    if result.shape[2] == 4:
                        opencv_mask = result[:, :, 3] / 255.0
                    else:
                        opencv_mask = np.where(np.all(result == [255, 255, 255], axis=2), 0, 1)
                    
                    opencv_coverage = np.mean(opencv_mask)
                    
                    if dl_coverage > opencv_coverage * 1.2:
                        logger.info(f"Using deep learning ({dl_coverage:.3f} vs {opencv_coverage:.3f})")
                        result = dl_result
                        method_used = f"Enhanced {dl_model}"
                        processing_time = dl_time
                    else:
                        logger.info(f"OpenCV better ({opencv_coverage:.3f} vs {dl_coverage:.3f})")
                        
                except Exception as e:
                    logger.info(f"Deep learning failed: {e}, using OpenCV result")
            
            return result, method_used, processing_time
            
        except Exception as e:
            logger.error(f"All background removal methods failed: {e}")
            return image, "No processing (failed)", 0.0


model_manager = ModelManager()
bg_remover = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bg_remover
    
    logger.info("Starting Background Removal API...")
    
    await model_manager.download_models()
    model_manager.load_models()
    bg_remover = BackgroundRemover(model_manager)
    
    logger.info("API ready for processing")
    
    yield
    
    logger.info("Shutting down...")

app = FastAPI(title="Background Removal API", version="1.0", lifespan=lifespan)

@app.get("/")
def read_root():
    return {
        "message": "Background Removal API",
        "version": "1.0",
        "models_loaded": list(model_manager.models.keys()) if model_manager.models else [],
        "device": str(device),
        "features": [
            "MODNet support",
            "Advanced matting",
            "Edge refinement",
            "Professional quality output"
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
            "memory_available": f"{memory_info.available / (1024**3):.1f}GB",
            "cpu_count": psutil.cpu_count()
        },
        "torch_info": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }

@app.post("/segmentation")
async def segment_image(challenge: str = Form(...), input: UploadFile = File(...)):
    start_time = datetime.now()
    
    try:
        logger.info(f"New request - Challenge: {challenge}, File: {input.filename}")
        
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
        
        logger.info(f"File saved: {len(contents)} bytes")
        
        image = cv2.imread(file_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read image")
        
        original_height, original_width = image.shape[:2]
        logger.info(f"Processing: {original_width}x{original_height}")
        
        result, method_used, processing_time_seconds = await bg_remover.remove_background(image)
        
        result_path = os.path.join(RESULT_DIR, f"{file_id}_processed{file_extension}")
        
        if result.shape[2] == 4:
            result_path = result_path.replace(file_extension, ".png")
            cv2.imwrite(result_path, result)
        else:
            cv2.imwrite(result_path, result)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"GPU memory cleaned. Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
        
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
                    "version": "1.0"
                }
                collection.insert_one(document)
                logger.info("Document stored in MongoDB")
            except Exception as e:
                logger.warning(f"MongoDB storage failed: {e}")
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Processing completed in {total_time:.2f}s")
        logger.info(f"Method: {method_used}, Processing: {processing_time_seconds:.2f}s")
        
        return JSONResponse(content={
            "message": "succeed", 
            "file_id": file_id,
            "method_used": method_used,
            "processing_time_seconds": processing_time_seconds,
            "total_time_seconds": total_time,
            "models_available": list(model_manager.models.keys()),
            "version": "1.0"
        })
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Processing failed: {str(e)}"}
        )

@app.get("/result/{file_id}")
async def get_result(file_id: str):
    for ext in ['.png', '.jpg', '.jpeg']:
        result_path = os.path.join(RESULT_DIR, f"{file_id}_processed{ext}")
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
        "model_configs": model_manager.model_configs,
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Background Removal API")
    logger.info(f"Device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")
    uvicorn.run(app, host="0.0.0.0", port=8080)
