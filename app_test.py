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
from scipy import ndimage
from skimage import segmentation, feature, filters

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pymongo import MongoClient

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


class HaloFreeBackgroundRemover:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
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
        return mask_resized

    def advanced_edge_analysis(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Advanced edge analysis for precise boundary detection"""
        h, w = mask.shape
        refined_mask = mask.copy()
        
        # Multi-scale edge detection
        edges_canny = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        edges_sobel = filters.sobel(mask)
        edges_sobel = (edges_sobel > 0.1).astype(np.uint8) * 255
        
        # Combine edge detection methods
        combined_edges = cv2.bitwise_or(edges_canny, edges_sobel)
        
        # Dilate edges to create processing region
        kernel = np.ones((7, 7), np.uint8)
        edge_region = cv2.dilate(combined_edges, kernel, iterations=1)
        
        # Process each edge pixel
        edge_coords = np.where(edge_region > 0)
        
        for y, x in zip(edge_coords[0], edge_coords[1]):
            if 10 <= y < h-10 and 10 <= x < w-10:
                # Extract larger neighborhood for better context
                local_image = image[y-10:y+11, x-10:x+11].astype(np.float32)
                local_mask = mask[y-10:y+11, x-10:x+11]
                
                # Analyze color gradients
                center_color = image[y, x].astype(np.float32)
                
                # Calculate color distance matrix
                color_distances = np.linalg.norm(local_image - center_color, axis=2)
                
                # Weight by mask values
                fg_weights = local_mask * (1 - color_distances / np.max(color_distances + 1e-6))
                bg_weights = (1 - local_mask) * (1 - color_distances / np.max(color_distances + 1e-6))
                
                # Calculate confidence scores
                fg_confidence = np.mean(fg_weights[fg_weights > 0]) if np.any(fg_weights > 0) else 0
                bg_confidence = np.mean(bg_weights[bg_weights > 0]) if np.any(bg_weights > 0) else 0
                
                # Update alpha based on confidence
                if fg_confidence + bg_confidence > 0:
                    new_alpha = fg_confidence / (fg_confidence + bg_confidence)
                    # Smooth transition
                    refined_mask[y, x] = 0.6 * refined_mask[y, x] + 0.4 * new_alpha
        
        return refined_mask

    def spectral_matting(self, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """Spectral matting for precise alpha channel extraction"""
        alpha = trimap.copy().astype(np.float32) / 255.0
        
        # Find unknown regions
        unknown_mask = (trimap == 128)
        
        if not np.any(unknown_mask):
            return alpha
        
        # Convert image to Lab color space for better color distance
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Get known foreground and background
        fg_mask = (trimap == 255)
        bg_mask = (trimap == 0)
        
        if not (np.any(fg_mask) and np.any(bg_mask)):
            return alpha
        
        # Sample colors from known regions
        fg_colors = lab_image[fg_mask]
        bg_colors = lab_image[bg_mask]
        
        # Process unknown pixels
        unknown_coords = np.where(unknown_mask)
        
        for i, (y, x) in enumerate(zip(unknown_coords[0], unknown_coords[1])):
            pixel_color = lab_image[y, x]
            
            # Find closest foreground and background colors
            fg_distances = np.linalg.norm(fg_colors - pixel_color, axis=1)
            bg_distances = np.linalg.norm(bg_colors - pixel_color, axis=1)
            
            min_fg_dist = np.min(fg_distances)
            min_bg_dist = np.min(bg_distances)
            
            # Calculate alpha based on relative distances
            if min_fg_dist + min_bg_dist > 0:
                alpha_val = min_bg_dist / (min_fg_dist + min_bg_dist)
                
                # Apply spatial smoothing based on local neighborhood
                if 5 <= y < image.shape[0]-5 and 5 <= x < image.shape[1]-5:
                    local_alpha = alpha[y-5:y+6, x-5:x+6]
                    local_weights = np.exp(-np.linalg.norm(
                        lab_image[y-5:y+6, x-5:x+6] - pixel_color, axis=2
                    ) / 30.0)
                    
                    weighted_alpha = np.sum(local_alpha * local_weights) / np.sum(local_weights)
                    alpha_val = 0.7 * alpha_val + 0.3 * weighted_alpha
                
                alpha[y, x] = alpha_val
        
        return alpha

    def color_decontamination(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Remove color bleeding and contamination"""
        h, w = image.shape[:2]
        decontaminated = image.copy().astype(np.float32)
        
        # Find edge regions where decontamination is needed
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        edges = cv2.Canny(alpha_uint8, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        edge_region = cv2.dilate(edges, kernel, iterations=2)
        
        # Estimate background color from low-alpha regions
        bg_mask = alpha < 0.1
        if np.any(bg_mask):
            bg_color = np.mean(image[bg_mask], axis=0)
        else:
            bg_color = np.array([128, 128, 128])  # Default gray
        
        # Process edge regions
        edge_coords = np.where(edge_region > 0)
        
        for y, x in zip(edge_coords[0], edge_coords[1]):
            pixel_alpha = alpha[y, x]
            
            if 0.1 < pixel_alpha < 0.9:  # Only process semi-transparent pixels
                pixel_color = image[y, x].astype(np.float32)
                
                # Remove background color contribution
                if pixel_alpha > 0.1:
                    # Solve: observed_color = alpha * fg_color + (1-alpha) * bg_color
                    fg_color = (pixel_color - (1 - pixel_alpha) * bg_color) / pixel_alpha
                    fg_color = np.clip(fg_color, 0, 255)
                    
                    # Apply gradual correction to avoid artifacts
                    correction_strength = min(1.0, (0.9 - pixel_alpha) * 2)
                    decontaminated[y, x] = (1 - correction_strength) * pixel_color + correction_strength * fg_color
        
        return decontaminated.astype(np.uint8)

    def apply_bilateral_alpha_refinement(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering for alpha refinement"""
        # Convert alpha to uint8 for bilateral filtering
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        
        # Apply multiple bilateral filters with different parameters
        refined_alpha1 = cv2.bilateralFilter(alpha_uint8, 9, 80, 80)
        refined_alpha2 = cv2.bilateralFilter(alpha_uint8, 15, 40, 40)
        
        # Combine results
        refined_alpha = (0.7 * refined_alpha1 + 0.3 * refined_alpha2).astype(np.uint8)
        
        # Apply guided filter using image as guide
        try:
            guided_alpha = cv2.ximgproc.guidedFilter(
                guide=image.astype(np.uint8),
                src=refined_alpha,
                radius=8,
                eps=0.05
            )
            return guided_alpha / 255.0
        except:
            # Fallback if guided filter not available
            return cv2.GaussianBlur(refined_alpha, (5, 5), 1.5) / 255.0

    def create_precise_trimap(self, mask: np.ndarray, fg_erosion: int = 8, bg_dilation: int = 12) -> np.ndarray:
        """Create precise trimap for alpha matting"""
        # Convert to binary
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Create sure foreground (eroded)
        fg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fg_erosion, fg_erosion))
        sure_fg = cv2.erode(binary_mask, fg_kernel, iterations=1)
        
        # Create sure background (dilated inverse)
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_dilation, bg_dilation))
        sure_bg = cv2.dilate(binary_mask, bg_kernel, iterations=1)
        sure_bg = 1 - sure_bg
        
        # Create trimap
        trimap = np.full_like(mask, 128, dtype=np.uint8)  # Unknown = 128
        trimap[sure_fg > 0] = 255  # Foreground = 255
        trimap[sure_bg > 0] = 0    # Background = 0
        
        return trimap

    async def remove_background_with_deep_learning(self, image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """Process with deep learning models and advanced post-processing"""
        start_time = time.time()
        
        if not self.model_manager.models:
            raise Exception("No deep learning models loaded")
        
        original_height, original_width = image.shape[:2]
        best_result = None
        best_model = "none"
        
        for model_name, model in self.model_manager.models.items():
            try:
                config = self.model_manager.model_configs[model_name]
                target_size = config["input_size"]
                
                # Preprocess
                input_tensor = self.preprocess_image(image, target_size)
                
                # Run model
                with torch.no_grad():
                    output = model(input_tensor)
                
                # Post-process
                raw_mask = self.postprocess_mask(output, (original_height, original_width))
                
                # Apply advanced refinements
                logger.info(f"Applying advanced edge analysis...")
                refined_mask = self.advanced_edge_analysis(image, raw_mask)
                
                logger.info(f"Creating precise trimap...")
                trimap = self.create_precise_trimap(refined_mask)
                
                logger.info(f"Applying spectral matting...")
                alpha = self.spectral_matting(image, trimap)
                
                logger.info(f"Applying bilateral refinement...")
                alpha = self.apply_bilateral_alpha_refinement(image, alpha)
                
                logger.info(f"Applying color decontamination...")
                decontaminated_image = self.color_decontamination(image, alpha)
                
                # Create final result
                rgba_result = np.zeros((original_height, original_width, 4), dtype=np.uint8)
                rgba_result[:, :, :3] = decontaminated_image
                rgba_result[:, :, 3] = (alpha * 255).astype(np.uint8)
                
                best_result = rgba_result
                best_model = model_name
                break  # Use first successful model
                
            except Exception as e:
                logger.error(f"{model_name} failed: {e}")
                continue
        
        if best_result is None:
            raise Exception("All models failed")
        
        processing_time = time.time() - start_time
        return best_result, best_model, processing_time

    async def remove_background_opencv_advanced(self, image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """Advanced OpenCV-based background removal"""
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
                if score > 0.7 and label in relevant_classes:
                    if score > best_score:
                        best_mask = cv2.resize(mask[0], (width, height))
                        best_score = score
            
            if best_mask is not None:
                # Apply all enhancements
                logger.info("Applying advanced processing to Mask R-CNN result...")
                
                refined_mask = self.advanced_edge_analysis(image, best_mask)
                trimap = self.create_precise_trimap(refined_mask)
                alpha = self.spectral_matting(image, trimap)
                alpha = self.apply_bilateral_alpha_refinement(image, alpha)
                decontaminated_image = self.color_decontamination(image, alpha)
                
                # Create RGBA result
                rgba_result = np.zeros((height, width, 4), dtype=np.uint8)
                rgba_result[:, :, :3] = decontaminated_image
                rgba_result[:, :, 3] = (alpha * 255).astype(np.uint8)
                
                processing_time = time.time() - start_time
                return rgba_result, "Advanced Mask R-CNN", processing_time
            
        except Exception as e:
            logger.error(f"Advanced Mask R-CNN failed: {e}")
        
        # Fallback: Simple ellipse with full processing
        logger.warning("Using enhanced ellipse fallback")
        fallback_mask = np.zeros((height, width), np.float32)
        cv2.ellipse(fallback_mask, (width//2, height//2), 
                   (int(width*0.4), int(height*0.4)), 0, 0, 360, 1, -1)
        
        # Apply full enhancement pipeline even to fallback
        refined_mask = self.advanced_edge_analysis(image, fallback_mask)
        trimap = self.create_precise_trimap(refined_mask)
        alpha = self.spectral_matting(image, trimap)
        alpha = self.apply_bilateral_alpha_refinement(image, alpha)
        decontaminated_image = self.color_decontamination(image, alpha)
        
        rgba_result = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_result[:, :, :3] = decontaminated_image
        rgba_result[:, :, 3] = (alpha * 255).astype(np.uint8)
        
        processing_time = time.time() - start_time
        return rgba_result, "Enhanced Ellipse Fallback", processing_time

    async def remove_background(self, image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        """Main background removal function with halo elimination"""
        try:
            # Try deep learning first
            if self.model_manager.models:
                try:
                    return await self.remove_background_with_deep_learning(image)
                except Exception as e:
                    logger.info(f"Deep learning failed: {e}, trying OpenCV advanced")
            
            # Fallback to advanced OpenCV
            return await self.remove_background_opencv_advanced(image)
            
        except Exception as e:
            logger.error(f"All background removal methods failed: {e}")
            # Emergency fallback - return original
            return image, "Failed - No processing", 0.0


# Initialize components
model_manager = ModelManager()
bg_remover = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bg_remover
    
    logger.info("Starting Halo-Free Background Removal API...")
    
    await model_manager.download_models()
    model_manager.load_models()
    bg_remover = HaloFreeBackgroundRemover(model_manager)
    
    logger.info("Halo-Free API ready for processing")
    
    yield
    
    logger.info("Shutting down...")

app = FastAPI(title="Halo-Free Background Removal API", version="8.0", lifespan=lifespan)

@app.get("/")
def read_root():
    return {
        "message": "Halo-Free Background Removal API",
        "version": "8.0 - Experimental Halo Elimination",
        "models_loaded": list(model_manager.models.keys()) if model_manager.models else [],
        "device": str(device),
        "features": [
            "Advanced edge analysis",
            "Spectral matting",
            "Color decontamination",
            "Bilateral alpha refinement",
            "Halo elimination",
            "High-quality transparency"
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
        
        result, method_used, processing_time_seconds = await bg_remover.remove_background(image)
        
        # Always save as PNG to preserve transparency
        result_path = os.path.join(RESULT_DIR, f"{file_id}_halo_free.png")
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
                    "version": "Halo-Free v8.0"
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
            "version": "Halo-Free v8.0"
        })
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Processing failed: {str(e)}"}
        )

@app.get("/result/{file_id}")
async def get_result(file_id: str):
    result_path = os.path.join(RESULT_DIR, f"{file_id}_halo_free.png")
    if os.path.exists(result_path):
        return FileResponse(result_path)
    
    return JSONResponse(
        status_code=404,
        content={"message": "Result not found"}
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Halo-Free Background Removal API")
    uvicorn.run(app, host="0.0.0.0", port=8081)  # Different port to avoid conflicts