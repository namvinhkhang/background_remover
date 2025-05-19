from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import cv2
import numpy as np
import os
from pymongo import MongoClient
import uuid
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Non-PyTorch Background Removal API")

# Create directories
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# MongoDB connection (optional)
client = None
db = None
collection = None

try:
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=2000)
    db = client["segmentation_db"]
    collection = db["images"]
    client.admin.command('ping')
    logger.info("✓ MongoDB connected")
except:
    logger.warning("✗ MongoDB not available - continuing without database")
    client = None

# Skip PyTorch entirely to avoid NumPy conflicts
PYTORCH_AVAILABLE = False
logger.info("ℹ️  Skipping PyTorch to avoid NumPy conflicts - using OpenCV only")

# Check sklearn
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
    logger.info("✓ Scikit-learn available")
except:
    SKLEARN_AVAILABLE = False
    logger.warning("✗ Scikit-learn not available")

class OpenCVBackgroundRemover:
    """Pure OpenCV background removal - no PyTorch conflicts"""
    
    def __init__(self):
        logger.info("Initialized Pure OpenCV background remover")
    
    def segment_with_watershed(self, img):
        """Advanced watershed segmentation"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Noise removal
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Threshold to get binary image
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Noise removal
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
            
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            _, markers = cv2.connectedComponents(sure_fg)
            
            # Add 1 to all labels so that sure background is not 0, but 1
            markers = markers + 1
            
            # Mark the region of unknown with zero
            markers[unknown == 255] = 0
            
            # Apply watershed
            markers = cv2.watershed(img, markers)
            
            # Create mask (everything except background)
            mask = np.where(markers > 1, 1, 0).astype(np.uint8)
            
            # Post-process
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Fill holes
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.fillPoly(mask, [largest_contour], 1)
            
            coverage = np.mean(mask)
            logger.info(f"Watershed segmentation coverage: {coverage:.3f}")
            return mask
            
        except Exception as e:
            logger.error(f"Watershed segmentation failed: {e}")
            return None
    
    def segment_with_mean_shift(self, img):
        """Mean shift segmentation"""
        try:
            # Apply mean shift filtering
            shifted = cv2.pyrMeanShiftFiltering(img, 20, 45)
            
            # Convert to grayscale and threshold
            gray_shifted = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_shifted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Check if we need to invert
            height, width = thresh.shape
            center_region = thresh[height//3:2*height//3, width//3:2*width//3]
            if np.mean(center_region) < 127:
                thresh = 255 - thresh
            
            # Clean up
            kernel = np.ones((5, 5), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Find largest component
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros(thresh.shape, np.uint8)
                cv2.fillPoly(mask, [largest_contour], 255)
                
                coverage = np.mean(mask) / 255.0
                logger.info(f"Mean shift segmentation coverage: {coverage:.3f}")
                return mask // 255
            
            return None
            
        except Exception as e:
            logger.error(f"Mean shift segmentation failed: {e}")
            return None
    
    def segment_with_ultra_grabcut(self, img):
        """Ultra-aggressive GrabCut with maximum inclusion"""
        try:
            height, width = img.shape[:2]
            
            # Method 1: Tiny margins for maximum inclusion
            margin_x = max(1, int(width * 0.002))   # 0.2% margin
            margin_y = max(1, int(height * 0.002))  # 0.2% margin
            rect1 = (margin_x, margin_y, width - 2*margin_x, height - 2*margin_y)
            
            mask1 = np.zeros((height, width), np.uint8)
            bgdModel1 = np.zeros((1, 65), np.float64)
            fgdModel1 = np.zeros((1, 65), np.float64)
            
            cv2.grabCut(img, mask1, rect1, bgdModel1, fgdModel1, 10, cv2.GC_INIT_WITH_RECT)
            result_mask1 = np.where((mask1 == 2) | (mask1 == 0), 0, 1).astype('uint8')
            
            # Method 2: Top region for hats/headphones
            top_rect = (0, 0, width, int(height * 0.7))
            mask_top = np.zeros((height, width), np.uint8)
            bgdModel_top = np.zeros((1, 65), np.float64)
            fgdModel_top = np.zeros((1, 65), np.float64)
            
            try:
                cv2.grabCut(img, mask_top, top_rect, bgdModel_top, fgdModel_top, 5, cv2.GC_INIT_WITH_RECT)
                result_mask_top = np.where((mask_top == 2) | (mask_top == 0), 0, 1).astype('uint8')
            except:
                result_mask_top = np.zeros((height, width), np.uint8)
            
            # Method 3: Left region for paws/arms
            left_rect = (0, int(height * 0.15), int(width * 0.65), int(height * 0.7))
            mask_left = np.zeros((height, width), np.uint8)
            bgdModel_left = np.zeros((1, 65), np.float64)
            fgdModel_left = np.zeros((1, 65), np.float64)
            
            try:
                cv2.grabCut(img, mask_left, left_rect, bgdModel_left, fgdModel_left, 5, cv2.GC_INIT_WITH_RECT)
                result_mask_left = np.where((mask_left == 2) | (mask_left == 0), 0, 1).astype('uint8')
            except:
                result_mask_left = np.zeros((height, width), np.uint8)
            
            # Method 4: Right region for paws/arms  
            right_rect = (int(width * 0.35), int(height * 0.15), int(width * 0.65), int(height * 0.7))
            mask_right = np.zeros((height, width), np.uint8)
            bgdModel_right = np.zeros((1, 65), np.float64)
            fgdModel_right = np.zeros((1, 65), np.float64)
            
            try:
                cv2.grabCut(img, mask_right, right_rect, bgdModel_right, fgdModel_right, 5, cv2.GC_INIT_WITH_RECT)
                result_mask_right = np.where((mask_right == 2) | (mask_right == 0), 0, 1).astype('uint8')
            except:
                result_mask_right = np.zeros((height, width), np.uint8)
            
            # Combine all masks
            combined_mask = np.maximum(result_mask1, result_mask_top)
            combined_mask = np.maximum(combined_mask, result_mask_left)
            combined_mask = np.maximum(combined_mask, result_mask_right)
            
            # Ultra-aggressive post-processing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
            
            # Maximum dilation for accessories
            kernel_dilate = np.ones((30, 30), np.uint8)
            combined_mask = cv2.dilate(combined_mask, kernel_dilate, iterations=4)
            
            # Fill all holes
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                for contour in contours:
                    cv2.fillPoly(combined_mask, [contour], 1)
            
            coverage = np.mean(combined_mask)
            logger.info(f"Ultra GrabCut coverage: {coverage:.3f}")
            return combined_mask
            
        except Exception as e:
            logger.error(f"Ultra GrabCut failed: {e}")
            return None
    
    def segment_with_edge_detection(self, img):
        """Enhanced edge-based segmentation"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Multiple edge detection methods
            edges_canny = cv2.Canny(blurred, 50, 150)
            edges_sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            edges_sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
            edges_sobel = np.uint8(edges_sobel / edges_sobel.max() * 255)
            
            # Combine edge maps
            combined_edges = cv2.bitwise_or(edges_canny, edges_sobel)
            
            # Dilate edges to create regions
            kernel = np.ones((5, 5), np.uint8)
            dilated_edges = cv2.dilate(combined_edges, kernel, iterations=2)
            
            # Find contours and create mask
            contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Create mask from all significant contours
                mask = np.zeros(gray.shape, np.uint8)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    # Include large contours
                    if area > (gray.shape[0] * gray.shape[1]) * 0.01:  # At least 1% of image
                        cv2.fillPoly(mask, [contour], 255)
                
                # Post-process
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
                
                coverage = np.mean(mask) / 255.0
                logger.info(f"Edge detection coverage: {coverage:.3f}")
                return mask // 255
            
            return None
            
        except Exception as e:
            logger.error(f"Edge detection failed: {e}")
            return None
    
    def segment_with_color_clustering(self, img):
        """Enhanced color clustering without sklearn dependency"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            height, width = img.shape[:2]
            
            # Sample from center region
            center_h, center_w = height//2, width//2
            sample_size = min(height, width) // 6
            
            center_region_bgr = img[center_h-sample_size:center_h+sample_size, 
                                  center_w-sample_size:center_w+sample_size]
            center_region_hsv = hsv[center_h-sample_size:center_h+sample_size,
                                  center_w-sample_size:center_w+sample_size]
            center_region_lab = lab[center_h-sample_size:center_h+sample_size,
                                  center_w-sample_size:center_w+sample_size]
            
            if center_region_bgr.size > 0:
                # Calculate mean colors
                mean_bgr = np.mean(center_region_bgr.reshape(-1, 3), axis=0)
                mean_hsv = np.mean(center_region_hsv.reshape(-1, 3), axis=0)
                mean_lab = np.mean(center_region_lab.reshape(-1, 3), axis=0)
                
                # Create color masks with generous ranges
                # BGR mask
                bgr_lower = np.array([max(0, mean_bgr[0]-60), max(0, mean_bgr[1]-60), max(0, mean_bgr[2]-60)])
                bgr_upper = np.array([min(255, mean_bgr[0]+60), min(255, mean_bgr[1]+60), min(255, mean_bgr[2]+60)])
                bgr_mask = cv2.inRange(img, bgr_lower, bgr_upper)
                
                # HSV mask
                hsv_lower = np.array([max(0, mean_hsv[0]-40), max(0, mean_hsv[1]-80), max(0, mean_hsv[2]-80)])
                hsv_upper = np.array([min(179, mean_hsv[0]+40), min(255, mean_hsv[1]+80), min(255, mean_hsv[2]+80)])
                hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
                
                # LAB mask
                lab_lower = np.array([max(0, mean_lab[0]-40), max(0, mean_lab[1]-40), max(0, mean_lab[2]-40)])
                lab_upper = np.array([min(255, mean_lab[0]+40), min(255, mean_lab[1]+40), min(255, mean_lab[2]+40)])
                lab_mask = cv2.inRange(lab, lab_lower, lab_upper)
                
                # Combine all masks
                combined_mask = cv2.bitwise_or(bgr_mask, hsv_mask)
                combined_mask = cv2.bitwise_or(combined_mask, lab_mask)
                
                # Aggressive expansion
                kernel = np.ones((25, 25), np.uint8)
                combined_mask = cv2.dilate(combined_mask, kernel, iterations=4)
                
                # Clean up
                kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_clean, iterations=3)
                
                # Fill holes
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    final_mask = np.zeros_like(combined_mask)
                    cv2.fillPoly(final_mask, [largest_contour], 255)
                    
                    coverage = np.mean(final_mask) / 255.0
                    logger.info(f"Color clustering coverage: {coverage:.3f}")
                    return final_mask // 255
            
            return None
            
        except Exception as e:
            logger.error(f"Color clustering failed: {e}")
            return None
    
    def create_premium_alpha(self, img, mask):
        """Create premium alpha channel with multiple smoothing techniques"""
        try:
            alpha = mask.astype(np.float32)
            
            # Multi-pass bilateral filtering for smooth edges
            for d, sigma_color, sigma_space in [(5, 40, 40), (9, 80, 80), (13, 120, 120)]:
                alpha_8bit = (alpha * 255).astype(np.uint8)
                alpha_filtered = cv2.bilateralFilter(alpha_8bit, d, sigma_color, sigma_space)
                alpha = alpha_filtered.astype(np.float32) / 255.0
            
            # Gaussian blur for additional smoothing
            alpha = cv2.GaussianBlur(alpha, (3, 3), 0.8)
            
            # Edge-preserving details
            # Find edges in original image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200) / 255.0
            
            # Reduce alpha near strong edges for crisp boundaries
            alpha_with_edges = alpha * (1 - 0.2 * edges)
            
            # Final gamma correction for natural look
            alpha_final = np.power(alpha_with_edges, 0.8)
            alpha_final = np.clip(alpha_final, 0.0, 1.0)
            
            return alpha_final
            
        except Exception as e:
            logger.error(f"Premium alpha creation failed: {e}")
            return mask.astype(np.float32)
    
    def remove_background(self, img):
        """Main background removal using pure OpenCV methods"""
        logger.info("=== Starting Pure OpenCV background removal ===")
        
        mask = None
        method_used = "unknown"
        
        # Method 1: Ultra GrabCut (most reliable)
        if mask is None:
            mask = self.segment_with_ultra_grabcut(img)
            if mask is not None:
                method_used = "Ultra GrabCut"
                coverage = np.mean(mask)
                logger.info(f"Using {method_used}, coverage: {coverage:.3f}")
                # Accept if reasonable coverage
                if coverage < 0.05:
                    mask = None
        
        # Method 2: Watershed segmentation
        if mask is None:
            mask = self.segment_with_watershed(img)
            if mask is not None:
                method_used = "Watershed Segmentation"
                coverage = np.mean(mask)
                logger.info(f"Using {method_used}, coverage: {coverage:.3f}")
                if coverage < 0.03:
                    mask = None
        
        # Method 3: Mean shift segmentation
        if mask is None:
            mask = self.segment_with_mean_shift(img)
            if mask is not None:
                method_used = "Mean Shift Segmentation"
                coverage = np.mean(mask)
                logger.info(f"Using {method_used}, coverage: {coverage:.3f}")
                if coverage < 0.03:
                    mask = None
        
        # Method 4: Color clustering
        if mask is None:
            mask = self.segment_with_color_clustering(img)
            if mask is not None:
                method_used = "Color Clustering"
                coverage = np.mean(mask)
                logger.info(f"Using {method_used}, coverage: {coverage:.3f}")
                if coverage < 0.02:
                    mask = None
        
        # Method 5: Edge detection
        if mask is None:
            mask = self.segment_with_edge_detection(img)
            if mask is not None:
                method_used = "Edge Detection"
                coverage = np.mean(mask)
                logger.info(f"Using {method_used}, coverage: {coverage:.3f}")
        
        # Method 6: Ultimate fallback - maximum ellipse
        if mask is None or np.mean(mask) < 0.01:
            logger.warning("Using maximum coverage ellipse fallback")
            height, width = img.shape[:2]
            mask = np.zeros((height, width), np.uint8)
            # Very large ellipse to ensure inclusion
            cv2.ellipse(mask, (width//2, height//2), 
                       (int(width*0.47), int(height*0.47)), 0, 0, 360, 1, -1)
            method_used = "Maximum ellipse fallback"
        
        # Final post-processing
        if mask is not None:
            # Very light cleaning to preserve details
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Gentle closing to connect nearby parts
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            
            # Final expansion for accessories
            kernel_expand = np.ones((6, 6), np.uint8)
            mask = cv2.dilate(mask, kernel_expand, iterations=2)
            
            # Comprehensive hole filling
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                for contour in contours:
                    cv2.fillPoly(mask, [contour], 1)
        
        final_coverage = np.mean(mask)
        logger.info(f"FINAL: {method_used}, coverage: {final_coverage:.3f}")
        
        # Create premium alpha channel
        alpha = self.create_premium_alpha(img, mask)
        
        # Apply alpha blending
        white_bg = np.ones_like(img, dtype=np.uint8) * 255
        alpha_3d = np.dstack([alpha] * 3)
        
        result = (alpha_3d * img.astype(np.float32) + 
                 (1 - alpha_3d) * white_bg.astype(np.float32)).astype(np.uint8)
        
        return result, method_used

# Initialize the OpenCV background remover
bg_remover = OpenCVBackgroundRemover()

@app.get("/")
def read_root():
    return {
        "message": "Pure OpenCV Background Removal API",
        "version": "5.0 - No PyTorch Conflicts",
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "sklearn_available": SKLEARN_AVAILABLE,
        "features": [
            "No PyTorch dependencies",
            "Ultra-aggressive GrabCut",
            "Watershed segmentation",
            "Mean shift clustering",
            "Advanced edge detection",
            "Premium alpha generation"
        ]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "pytorch_available": False,
        "sklearn_available": SKLEARN_AVAILABLE,
        "methods": [
            "Ultra GrabCut",
            "Watershed",
            "Mean Shift",
            "Color Clustering", 
            "Edge Detection"
        ]
    }

@app.post("/segmentation")
async def segment_image(challenge: str = Form(...), input: UploadFile = File(...)):
    """Pure OpenCV segmentation - no PyTorch conflicts"""
    start_time = datetime.now()
    
    try:
        logger.info(f"=== NEW OPENCV-ONLY REQUEST ===")
        logger.info(f"Challenge: {challenge}, File: {input.filename}")
        logger.info(f"NumPy version: {np.__version__}, OpenCV: {cv2.__version__}")
        
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
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(input.filename)[1] or ".jpg"
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
        
        contents = await input.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"File saved: {len(contents)} bytes")
        
        # Read and process image
        img = cv2.imread(file_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not read image")
        
        original_height, original_width = img.shape[:2]
        logger.info(f"Processing: {original_width}x{original_height}")
        
        # Remove background using pure OpenCV
        result, method_used = bg_remover.remove_background(img)
        
        # Save result
        result_path = os.path.join(RESULT_DIR, f"{file_id}_opencv{file_extension}")
        cv2.imwrite(result_path, result)
        
        # Store in MongoDB if available
        if collection is not None:
            try:
                processing_time = (datetime.now() - start_time).total_seconds()
                document = {
                    "original_file": file_path,
                    "processed_file": result_path,
                    "challenge": challenge,
                    "timestamp": start_time,
                    "file_id": file_id,
                    "original_size": f"{original_width}x{original_height}",
                    "method_used": method_used,
                    "processing_time_seconds": processing_time,
                    "numpy_version": np.__version__,
                    "opencv_version": cv2.__version__,
                    "version": "Pure OpenCV v5.0"
                }
                collection.insert_one(document)
                logger.info("Document stored in MongoDB")
            except Exception as e:
                logger.warning(f"MongoDB storage failed: {e}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"=== OPENCV PROCESSING COMPLETED in {processing_time:.2f}s ===")
        logger.info(f"Method used: {method_used}")
        
        return JSONResponse(content={
            "message": "succeed", 
            "file_id": file_id,
            "method_used": method_used,
            "processing_time_seconds": processing_time,
            "version": "Pure OpenCV - No Conflicts"
        })
        
    except Exception as e:
        logger.error(f"Pure OpenCV processing error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Processing failed: {str(e)}"}
        )

@app.get("/result/{file_id}")
async def get_result(file_id: str):
    """Get pure OpenCV result"""
    for ext in ['.jpg', '.jpeg', '.png']:
        result_path = os.path.join(RESULT_DIR, f"{file_id}_opencv{ext}")
        if os.path.exists(result_path):
            return FileResponse(result_path)
    
    return JSONResponse(
        status_code=404,
        content={"message": "Result not found"}
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Pure OpenCV Background Removal API")
    logger.info(f"🖼️  OpenCV version: {cv2.__version__}")
    logger.info(f"🔢 NumPy version: {np.__version__}")
    logger.info("✨ No PyTorch conflicts - pure computer vision!")
    uvicorn.run(app, host="0.0.0.0", port=8080)