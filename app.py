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
from typing import Optional, Tuple, List, Callable, Any
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

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

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from pymongo import MongoClient
from pydantic import BaseModel, validator, Field

from u2net import U2NET, U2NETP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
RESULT_DIR = os.getenv("RESULT_DIR", "results")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "models/pretrained")

class BoundaryLimits:
    CHALLENGE_MIN_LENGTH = 3
    CHALLENGE_MAX_LENGTH = 3
    PROCESSING_MODE_MIN_LENGTH = 4 
    PROCESSING_MODE_MAX_LENGTH = 9
    
    MIN_FILE_SIZE = 1024
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    MIN_IMAGE_WIDTH = 32
    MIN_IMAGE_HEIGHT = 32
    MAX_IMAGE_WIDTH = 10000
    MAX_IMAGE_HEIGHT = 10000
    
    MIN_PROCESSING_SIZE = 128
    MAX_PROCESSING_SIZE = 2048
    MIN_REFINEMENT_THRESHOLD = 0.1
    MAX_REFINEMENT_THRESHOLD = 100.0
    
    MIN_AVAILABLE_MEMORY_MB = 500
    MAX_PROCESSING_MEMORY_MB = 4000
    
    MAX_PROCESSING_TIME_SECONDS = 300
    MAX_FILE_IO_TIME_SECONDS = 30
    
    VALID_CHALLENGES = {"cv3"}
    VALID_PROCESSING_MODES = {"fast", "balanced", "quality"}
    VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    VALID_IMAGE_MIME_TYPES = {
        "image/jpeg", "image/jpg", "image/png", "image/bmp", 
        "image/tiff", "image/webp", "image/x-ms-bmp"
    }

class SecureTypeValidator:
    IMAGE_MAGIC_NUMBERS = {
        b'\xFF\xD8\xFF': 'JPEG',
        b'\x89PNG\r\n\x1a\n': 'PNG', 
        b'BM': 'BMP',
        b'GIF87a': 'GIF',
        b'GIF89a': 'GIF',
        b'RIFF': 'WEBP',
        b'\x00\x00\x01\x00': 'ICO',
        b'II*\x00': 'TIFF',
        b'MM\x00*': 'TIFF'
    }
    
    DANGEROUS_SIGNATURES = {
        b'MZ': 'Windows Executable (.exe, .dll)',
        b'\x7fELF': 'Linux Executable (ELF)',
        b'\xca\xfe\xba\xbe': 'Java Class File',
        b'PK\x03\x04': 'ZIP/JAR (could contain executables)',
        b'#!/bin/': 'Shell Script',
        b'#!/usr/bin/': 'Shell Script', 
        b'@echo off': 'Batch Script',
        b'<?xml': 'XML (could be malicious)',
        b'<script': 'JavaScript/HTML',
        b'<html': 'HTML content',
        b'\x1f\x8b': 'GZIP (compressed executable?)',
        b'\x50\x4b': 'ZIP archive',
        b'\x7f\x45\x4c\x46': 'ELF executable',
        b'\xd0\xcf\x11\xe0': 'Microsoft Office document',
        b'%PDF': 'PDF document'
    }
    
    @staticmethod
    def validate_file_signature(file_content: bytes) -> str:
        if not isinstance(file_content, bytes):
            raise ValueError("File content must be bytes")
        
        if len(file_content) < 12:
            raise ValueError("File too small to determine type")
        
        for signature, file_type in SecureTypeValidator.IMAGE_MAGIC_NUMBERS.items():
            if file_content.startswith(signature):
                if file_type == 'WEBP':
                    if len(file_content) > 12 and b'WEBP' in file_content[8:12]:
                        return file_type
                    else:
                        continue
                else:
                    return file_type
        
        raise ValueError("File is not a valid image - invalid file signature")
    
    @staticmethod
    def detect_dangerous_content(file_content: bytes) -> None:
        if not isinstance(file_content, bytes):
            raise ValueError("File content must be bytes")
        
        check_bytes = file_content[:1024]
        
        for signature, description in SecureTypeValidator.DANGEROUS_SIGNATURES.items():
            if signature in check_bytes:
                raise ValueError(f"Dangerous file detected: {description}")
        
        if b'<script' in check_bytes.lower():
            raise ValueError("HTML/JavaScript content detected")
        
        if b'#!/' in check_bytes:
            raise ValueError("Shell script detected")
        
        if b'exec' in check_bytes.lower() and b'(' in check_bytes:
            raise ValueError("Executable code pattern detected")
    
    @staticmethod
    def is_binary_file(file_content: bytes) -> bool:
        if not isinstance(file_content, bytes):
            raise ValueError("File content must be bytes")
        
        if len(file_content) == 0:
            return False
        
        if b'\x00' not in file_content[:1024]:
            return False
        
        sample_size = min(1024, len(file_content))
        text_chars = sum(1 for byte in file_content[:sample_size] 
                        if 32 <= byte <= 126 or byte in [9, 10, 13])
        text_ratio = text_chars / sample_size
        
        if text_ratio > 0.9:
            return False
            
        return True
    
    @staticmethod
    def validate_image_structure(file_content: bytes) -> Tuple[int, int, str]:
        if not isinstance(file_content, bytes):
            raise ValueError("File content must be bytes")
        
        try:
            image = Image.open(io.BytesIO(file_content))
            image.verify()
            
            image = Image.open(io.BytesIO(file_content))
            image.load()
            
            if image.mode not in ['RGB', 'RGBA', 'L', 'P', 'CMYK']:
                raise ValueError(f"Unsupported image mode: {image.mode}")
            
            width, height = image.size
            
            if width < BoundaryLimits.MIN_IMAGE_WIDTH or height < BoundaryLimits.MIN_IMAGE_HEIGHT:
                raise ValueError(f"Image too small: {width}x{height}, minimum: {BoundaryLimits.MIN_IMAGE_WIDTH}x{BoundaryLimits.MIN_IMAGE_HEIGHT}")
            
            if width > BoundaryLimits.MAX_IMAGE_WIDTH or height > BoundaryLimits.MAX_IMAGE_HEIGHT:
                raise ValueError(f"Image too large: {width}x{height}, maximum: {BoundaryLimits.MAX_IMAGE_WIDTH}x{BoundaryLimits.MAX_IMAGE_HEIGHT}")
            
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                for tag, value in exif.items():
                    if isinstance(value, str) and len(value) > 1000:
                        raise ValueError("Suspicious EXIF data detected")
            
            return width, height, image.mode
            
        except Exception as e:
            if "suspicious" in str(e).lower() or "dangerous" in str(e).lower():
                raise
            else:
                raise ValueError(f"Invalid image structure: {str(e)}")

class InputValidator:
    @staticmethod
    def sanitize_string(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Input must be a string")
        
        sanitized = value.strip()
        sanitized = sanitized.replace('\x00', '').replace('\n', '').replace('\r', '')
        sanitized = sanitized.replace('\t', '').replace('\b', '').replace('\f', '')
        
        return sanitized
    
    @staticmethod
    def validate_string_boundaries(value: str, field_name: str, min_length: int, max_length: int, 
                                 allowed_chars_pattern: str = None) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")
        
        if not isinstance(min_length, int) or not isinstance(max_length, int):
            raise ValueError("Length boundaries must be integers")
        
        if min_length < 0 or max_length < 0:
            raise ValueError("Length boundaries must be non-negative")
        
        if min_length > max_length:
            raise ValueError("Minimum length cannot be greater than maximum length")
        
        sanitized_value = InputValidator.sanitize_string(value)
        
        if len(sanitized_value) < min_length:
            raise ValueError(f"{field_name} must be at least {min_length} characters long")
        
        if len(sanitized_value) > max_length:
            raise ValueError(f"{field_name} must not exceed {max_length} characters")
        
        if allowed_chars_pattern and not re.match(allowed_chars_pattern, sanitized_value):
            raise ValueError(f"{field_name} contains invalid characters")
        
        return sanitized_value
    
    @staticmethod
    def validate_challenge(challenge: str) -> str:
        if not isinstance(challenge, str):
            raise ValueError("Challenge must be a string")
        
        sanitized = InputValidator.validate_string_boundaries(
            challenge, "challenge", 
            BoundaryLimits.CHALLENGE_MIN_LENGTH, 
            BoundaryLimits.CHALLENGE_MAX_LENGTH,
            r'^[a-zA-Z0-9]+$'
        )
        
        if sanitized not in BoundaryLimits.VALID_CHALLENGES:
            raise ValueError(f"Invalid challenge value. Allowed: {BoundaryLimits.VALID_CHALLENGES}")
        
        return sanitized
    
    @staticmethod
    def validate_processing_mode(mode: str) -> str:
        if not isinstance(mode, str):
            raise ValueError("Processing mode must be a string")
        
        sanitized = InputValidator.validate_string_boundaries(
            mode, "processing_mode",
            BoundaryLimits.PROCESSING_MODE_MIN_LENGTH,
            BoundaryLimits.PROCESSING_MODE_MAX_LENGTH,
            r'^[a-zA-Z0-9_-]+$'
        )
        
        if sanitized not in BoundaryLimits.VALID_PROCESSING_MODES:
            raise ValueError(f"Invalid processing mode. Allowed: {BoundaryLimits.VALID_PROCESSING_MODES}")
        
        return sanitized
    
    @staticmethod
    def validate_numeric_boundaries(value: any, field_name: str, min_val: float, max_val: float) -> float:
        if not isinstance(field_name, str):
            raise ValueError("Field name must be a string")
        
        if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
            raise ValueError("Boundary values must be numeric")
        
        if min_val > max_val:
            raise ValueError("Minimum value cannot be greater than maximum value")
        
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
        if not isinstance(field_name, str):
            raise ValueError("Field name must be a string")
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            sanitized = InputValidator.sanitize_string(value).lower()
            if sanitized in {'true', '1', 'yes', 'on'}:
                return True
            elif sanitized in {'false', '0', 'no', 'off'}:
                return False
        
        raise ValueError(f"{field_name} must be a valid boolean value")
    
    @staticmethod
    async def comprehensive_file_validation(file: UploadFile) -> Tuple[bytes, int, int, str]:
        if not file or not file.filename:
            raise ValueError("No file provided")
        
        sanitized_filename = InputValidator.sanitize_string(file.filename)
        if not sanitized_filename:
            raise ValueError("Invalid filename")
        
        try:
            file_content = await asyncio.wait_for(file.read(), timeout=BoundaryLimits.MAX_FILE_IO_TIME_SECONDS)
        except asyncio.TimeoutError:
            raise ValueError("File read timeout - file too large or slow connection")
        
        await file.seek(0)
        
        file_size = len(file_content)
        if file_size < BoundaryLimits.MIN_FILE_SIZE:
            raise ValueError(f"File too small. Minimum size: {BoundaryLimits.MIN_FILE_SIZE} bytes")
        
        if file_size > BoundaryLimits.MAX_FILE_SIZE:
            raise ValueError(f"File too large. Maximum size: {BoundaryLimits.MAX_FILE_SIZE} bytes "
                           f"({BoundaryLimits.MAX_FILE_SIZE // (1024*1024)}MB)")
        
        if not file.content_type or file.content_type not in BoundaryLimits.VALID_IMAGE_MIME_TYPES:
            raise ValueError(f"Invalid file type. Allowed types: {BoundaryLimits.VALID_IMAGE_MIME_TYPES}")
        
        file_ext = os.path.splitext(sanitized_filename)[1].lower()
        if file_ext not in BoundaryLimits.VALID_IMAGE_EXTENSIONS:
            raise ValueError(f"Invalid file extension. Allowed: {BoundaryLimits.VALID_IMAGE_EXTENSIONS}")
        
        detected_type = SecureTypeValidator.validate_file_signature(file_content)
        SecureTypeValidator.detect_dangerous_content(file_content)
        
        if not SecureTypeValidator.is_binary_file(file_content):
            raise ValueError("File appears to be text, not a valid image")
        
        width, height, mode = SecureTypeValidator.validate_image_structure(file_content)
        
        expected_extensions = {
            'JPEG': ['.jpg', '.jpeg'],
            'PNG': ['.png'],
            'BMP': ['.bmp'],
            'GIF': ['.gif'],
            'WEBP': ['.webp'],
            'TIFF': ['.tiff', '.tif'],
            'ICO': ['.ico']
        }
        
        if file_ext not in expected_extensions.get(detected_type, []):
            raise ValueError(f"File extension {file_ext} doesn't match actual file type {detected_type}")
        
        logger.info(f"File validation passed: {sanitized_filename}, size: {file_size} bytes, type: {detected_type}, dimensions: {width}x{height}")
        
        return file_content, width, height, detected_type

class ResourceValidator:
    @staticmethod
    def check_available_memory(required_mb: float) -> bool:
        if not isinstance(required_mb, (int, float)):
            raise ValueError("Required memory must be a number")
        
        if required_mb < 0:
            raise ValueError("Required memory cannot be negative")
        
        available = psutil.virtual_memory().available / (1024*1024)
        
        if available < BoundaryLimits.MIN_AVAILABLE_MEMORY_MB:
            raise ValueError(f"Insufficient system memory. Available: {available:.1f}MB, Minimum required: {BoundaryLimits.MIN_AVAILABLE_MEMORY_MB}MB")
        
        if required_mb > BoundaryLimits.MAX_PROCESSING_MEMORY_MB:
            raise ValueError(f"Requested memory too high: {required_mb:.1f}MB, Maximum allowed: {BoundaryLimits.MAX_PROCESSING_MEMORY_MB}MB")
        
        return available > (required_mb * 1.5)
    
    @staticmethod
    def estimate_processing_memory(width: int, height: int, channels: int = 3) -> float:
        if not isinstance(width, int) or not isinstance(height, int) or not isinstance(channels, int):
            raise ValueError("Image dimensions must be integers")
        
        if width <= 0 or height <= 0 or channels <= 0:
            raise ValueError("Image dimensions must be positive")
        
        if width > BoundaryLimits.MAX_IMAGE_WIDTH or height > BoundaryLimits.MAX_IMAGE_HEIGHT:
            raise ValueError(f"Image dimensions too large: {width}x{height}")
        
        base_memory_mb = (width * height * channels * 4) / (1024*1024)
        model_memory_mb = base_memory_mb * 8
        total_memory_mb = base_memory_mb + model_memory_mb
        
        return total_memory_mb
    
    @staticmethod
    def check_disk_space(path: str, required_mb: float) -> bool:
        if not isinstance(path, str):
            raise ValueError("Path must be a string")
        
        if not isinstance(required_mb, (int, float)):
            raise ValueError("Required space must be a number")
        
        if required_mb < 0:
            raise ValueError("Required space cannot be negative")
        
        if not os.path.exists(path):
            raise ValueError(f"Path does not exist: {path}")
        
        available_mb = psutil.disk_usage(path).free / (1024*1024)
        
        return available_mb > (required_mb * 1.2)

# VALIDATION DECORATORS
def validate_file_upload(func: Callable) -> Callable:
    """Decorator to validate file uploads with comprehensive security checks"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        input_file = None
        if 'input' in kwargs:
            input_file = kwargs['input']
        
        if input_file is None:
            return JSONResponse(
                status_code=400,
                content={
                    "message": "No file provided",
                    "error_type": "file_validation_error"
                }
            )
        
        try:
            file_content, width, height, detected_type = await InputValidator.comprehensive_file_validation(input_file)
            kwargs['validated_file_data'] = (file_content, width, height, detected_type)
            return await func(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"File validation failed: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"File validation failed: {str(e)}",
                    "error_type": "file_validation_error",
                    "boundary_limits": {
                        "max_file_size_mb": BoundaryLimits.MAX_FILE_SIZE // (1024*1024),
                        "min_file_size_kb": BoundaryLimits.MIN_FILE_SIZE // 1024,
                        "supported_formats": list(BoundaryLimits.VALID_IMAGE_EXTENSIONS),
                        "max_image_dimensions": f"{BoundaryLimits.MAX_IMAGE_WIDTH}x{BoundaryLimits.MAX_IMAGE_HEIGHT}"
                    }
                }
            )
    return wrapper

def validate_input_parameters(func: Callable) -> Callable:
    """Decorator to validate input parameters (challenge, processing_mode)"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            if 'challenge' in kwargs:
                kwargs['challenge'] = InputValidator.validate_challenge(kwargs['challenge'])
            
            if 'processing_mode' in kwargs:
                kwargs['processing_mode'] = InputValidator.validate_processing_mode(kwargs['processing_mode'])
            
            return await func(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Parameter validation failed: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"Parameter validation failed: {str(e)}",
                    "error_type": "parameter_validation_error",
                    "valid_challenges": list(BoundaryLimits.VALID_CHALLENGES),
                    "valid_processing_modes": list(BoundaryLimits.VALID_PROCESSING_MODES)
                }
            )
    return wrapper

def validate_system_resources(func: Callable) -> Callable:
    """Decorator to validate system resources before processing"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            if 'validated_file_data' in kwargs:
                file_content, width, height, detected_type = kwargs['validated_file_data']
                
                required_memory = ResourceValidator.estimate_processing_memory(width, height)
                if not ResourceValidator.check_available_memory(required_memory):
                    raise ValueError(f"Insufficient memory for processing. Required: {required_memory:.1f}MB")
                
                estimated_result_size = (width * height * 4) / (1024*1024)
                if not ResourceValidator.check_disk_space(RESULT_DIR, estimated_result_size):
                    raise ValueError("Insufficient disk space for saving results")
                
                kwargs['estimated_memory_usage'] = required_memory
            
            return await func(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Resource validation failed: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"Resource validation failed: {str(e)}",
                    "error_type": "resource_validation_error"
                }
            )
    return wrapper

def validate_file_id(func: Callable) -> Callable:
    """Decorator to validate file ID parameters"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            if 'file_id' in kwargs:
                kwargs['file_id'] = InputValidator.validate_string_boundaries(
                    kwargs['file_id'], "file_id", 1, 100, r'^[a-zA-Z0-9-]+$'
                )
            
            return await func(*args, **kwargs)
        except ValueError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"Invalid file ID: {str(e)}",
                    "error_type": "file_id_validation_error"
                }
            )
    return wrapper

def timeout_protection(timeout_seconds: int = BoundaryLimits.MAX_PROCESSING_TIME_SECONDS):
    """Decorator to add timeout protection to processing functions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.error(f"Processing timeout after {timeout_seconds} seconds")
                return JSONResponse(
                    status_code=400,
                    content={
                        "message": f"Processing timeout - exceeded {timeout_seconds} seconds",
                        "error_type": "timeout_error"
                    }
                )
        return wrapper
    return decorator

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
        if not isinstance(self.model_configs, dict):
            raise ValueError("Model configs must be a dictionary")
        
        for model_name, config in self.model_configs.items():
            if not isinstance(config, dict):
                raise ValueError(f"Model config for {model_name} must be a dictionary")
            
            if 'filename' not in config:
                raise ValueError(f"Model config for {model_name} missing filename")
            
            model_path = os.path.join(MODEL_CACHE_DIR, config["filename"])
            
            if not os.path.exists(model_path):
                logger.info(f"Downloading {model_name}")
                try:
                    if "drive.google.com" in config["url"]:
                        gdown.download(config["url"], model_path, quiet=False)
                    logger.info(f"{model_name} downloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to download {model_name}: {e}")
                    if model_name == "u2net" and os.path.exists("u2net.pth"):
                        import shutil
                        shutil.copy("u2net.pth", model_path)
            else:
                logger.info(f"{model_name} already exists")

    def load_u2net_model(self, model_name: str) -> Optional[torch.nn.Module]:
        if not isinstance(model_name, str):
            raise ValueError("Model name must be a string")
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model name: {model_name}")
        
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
        if not isinstance(model_manager, ModelManager):
            raise ValueError("Model manager must be an instance of ModelManager")
        
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        
        self.max_processing_size = InputValidator.validate_numeric_boundaries(
            512, "max_processing_size", 
            BoundaryLimits.MIN_PROCESSING_SIZE, BoundaryLimits.MAX_PROCESSING_SIZE
        )
        self.enable_refinement = True
        self.refinement_threshold = 2048 * 2048
        
    def preprocess_image_u2net_fixed(self, image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        
        if len(image.shape) != 3:
            raise ValueError("Image must be 3-dimensional (H, W, C)")
        
        if not isinstance(target_size, tuple) or len(target_size) != 2:
            raise ValueError("Target size must be a tuple of (width, height)")
        
        original_height, original_width = image.shape[:2]
        
        if original_width <= 0 or original_height <= 0:
            raise ValueError("Image dimensions must be positive")
        
        target_width, target_height = target_size
        if target_width <= 0 or target_height <= 0:
            raise ValueError("Target dimensions must be positive")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        if not isinstance(output, (torch.Tensor, tuple)):
            raise ValueError("Output must be a torch tensor or tuple")
        
        if not isinstance(original_shape, tuple) or len(original_shape) != 2:
            raise ValueError("Original shape must be a tuple of (height, width)")
        
        if not isinstance(padding_info, tuple) or len(padding_info) != 4:
            raise ValueError("Padding info must be a tuple of 4 values")
        
        if isinstance(output, tuple):
            mask = output[0]
        else:
            mask = output
            
        mask_np = mask.squeeze().cpu().detach().numpy()
        
        pad_y, pad_x, new_height, new_width = padding_info
        
        if pad_y < 0 or pad_x < 0 or new_height <= 0 or new_width <= 0:
            raise ValueError("Invalid padding information")
        
        mask_crop = mask_np[pad_y:pad_y + new_height, pad_x:pad_x + new_width]
        
        mask_resized = cv2.resize(mask_crop, (original_shape[1], original_shape[0]))
        
        mask_resized = np.clip(mask_resized, 0, 1)
        
        return mask_resized

    def ensemble_masks_improved(self, masks: list, confidences: list = None) -> np.ndarray:
        if not isinstance(masks, list):
            raise ValueError("Masks must be a list")
        
        if len(masks) == 0:
            raise ValueError("Masks list cannot be empty")
            
        if len(masks) == 1:
            return masks[0]
        
        if confidences is not None:
            if not isinstance(confidences, list):
                raise ValueError("Confidences must be a list")
            if len(confidences) != len(masks):
                raise ValueError("Confidences must have same length as masks")
        else:
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
        if not isinstance(mask, np.ndarray):
            raise ValueError("Mask must be a numpy array")
        
        if len(mask.shape) != 2:
            raise ValueError("Mask must be 2-dimensional")
        
        refinement_strength = InputValidator.validate_numeric_boundaries(
            refinement_strength, "refinement_strength", 0.0, 1.0
        )
        
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
        if not isinstance(image, np.ndarray) or not isinstance(coarse_mask, np.ndarray) or not isinstance(trimap, np.ndarray):
            raise ValueError("All inputs must be numpy arrays")
        
        if image.shape[:2] != coarse_mask.shape or image.shape[:2] != trimap.shape:
            raise ValueError("Image, mask, and trimap must have the same spatial dimensions")
        
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
        if not isinstance(image, np.ndarray) or not isinstance(alpha, np.ndarray):
            raise ValueError("Image and alpha must be numpy arrays")
        
        if image.shape[:2] != alpha.shape:
            raise ValueError("Image and alpha must have the same spatial dimensions")
        
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
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        
        if len(image.shape) != 3:
            raise ValueError("Image must be 3-dimensional")
        
        original_h, original_w = image.shape[:2]
        
        if original_w <= 0 or original_h <= 0:
            raise ValueError("Image dimensions must be positive")
        
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
        
        if not u2net_models:
            raise ValueError("No U2NET models available")
        
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
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        
        if not self.enable_refinement:
            return False
        
        h, w = image.shape[:2]
        return (h * w) > self.refinement_threshold

    async def refine_mask_with_guided_matting(self, image: np.ndarray, coarse_mask: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or not isinstance(coarse_mask, np.ndarray):
            raise ValueError("Image and mask must be numpy arrays")
        
        self.logger.info("Applying guided refinement on full resolution")
        
        trimap = self.create_guided_trimap(coarse_mask, refinement_strength=0.2)
        
        refined_alpha = self.guided_refinement_matting(image, coarse_mask, trimap)
        
        final_alpha = self.apply_edge_preserving_smoothing(image, refined_alpha)
        
        return final_alpha

    async def remove_background(self, image: np.ndarray) -> Tuple[np.ndarray, str, float]:
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        
        if len(image.shape) != 3:
            raise ValueError("Image must be 3-dimensional (H, W, C)")
        
        start_time = time.time()
        
        try:
            h, w = image.shape[:2]
            required_memory = ResourceValidator.estimate_processing_memory(w, h)
            
            if not ResourceValidator.check_available_memory(required_memory):
                raise ValueError(f"Insufficient memory for processing. Required: {required_memory:.1f}MB")
            
            self.logger.info("Getting coarse mask with global context")
            coarse_mask, coarse_method = await self.get_coarse_mask_with_context(image)
            
            if self.should_use_refinement(image):
                self.logger.info("Image is high-res, applying guided refinement")
                final_alpha = await self.refine_mask_with_guided_matting(image, coarse_mask)
                method = f"{coarse_method} + Guided-Refinement"
            else:
                self.logger.info("Using coarse mask directly")
                final_alpha = coarse_mask
                method = coarse_method
            
            if not isinstance(final_alpha, np.ndarray):
                raise ValueError("Final alpha must be a numpy array")
            
            if final_alpha.shape != (h, w):
                raise ValueError("Final alpha dimensions don't match image")
            
            rgba_result = np.zeros((h, w, 4), dtype=np.uint8)
            rgba_result[:, :, :3] = image
            rgba_result[:, :, 3] = (final_alpha * 255).astype(np.uint8)
            
            processing_time = time.time() - start_time
            
            if processing_time > BoundaryLimits.MAX_PROCESSING_TIME_SECONDS:
                self.logger.warning(f"Processing time exceeded limit: {processing_time:.2f}s > {BoundaryLimits.MAX_PROCESSING_TIME_SECONDS}s")
            
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
    if not isinstance(model_manager, ModelManager):
        raise ValueError("Model manager must be an instance of ModelManager")
    
    if not isinstance(config_name, str):
        raise ValueError("Config name must be a string")
    
    if config_name not in PROCESSING_CONFIGS:
        raise ValueError(f"Invalid config name. Available: {list(PROCESSING_CONFIGS.keys())}")
    
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
    
    logger.info("Starting HYBRID U2NET Background Removal API")
    
    await model_manager.download_models()
    model_manager.load_models()
    
    bg_remover = create_optimized_bg_remover(model_manager, "balanced")
    
    logger.info("API ready for processing")
    
    yield
    
    logger.info("Shutting down")

app = FastAPI(title="Background Removal API", version="2.0", lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content=jsonable_encoder({
            "message": "Wrong type of inputs or missing parameters.",
            "detail": "Need 'challenge' and 'input'. The 'challenge' parameter must be cv3 and the 'input' must be an image with the correct extension.",
        }),
    )

@app.post("/segmentation")
@validate_input_parameters
@validate_file_upload
@validate_system_resources
@timeout_protection()
async def segment_image(
    challenge: str = Form(...),
    input: UploadFile = File(...),
    processing_mode: str = Form("fast"),
    validated_file_data: tuple = None,
    estimated_memory_usage: float = None
):
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing request - Challenge: {challenge}, File: {input.filename}, Mode: {processing_mode}")
        
        file_content, width, height, detected_type = validated_file_data
        
        if processing_mode in PROCESSING_CONFIGS:
            global bg_remover
            bg_remover = create_optimized_bg_remover(model_manager, processing_mode)
        
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(input.filename)[1].lower() or ".jpg"
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        image = cv2.imread(file_path)
        if image is None:
            try:
                os.remove(file_path)
            except:
                pass
            raise HTTPException(status_code=400, detail="Could not read image file after validation")
        
        logger.info(f"Processing validated image: {width}x{height} ({(width*height) / (1024*1024):.1f}MP), type: {detected_type}")
        
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
                    "original_size": f"{width}x{height}",
                    "megapixels": (width * height) / (1024 * 1024),
                    "detected_file_type": detected_type,
                    "method_used": method_used,
                    "processing_mode": processing_mode,
                    "processing_time_seconds": processing_time_seconds,
                    "total_time_seconds": total_time,
                    "device": str(device),
                    "used_refinement": "Refinement" in method_used,
                    "file_size_bytes": len(file_content),
                    "boundary_validation_passed": True,
                    "memory_used_mb": estimated_memory_usage
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
        })
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=400,
            content={"message": f"Processing failed: {str(e)}"}
        )

@app.get("/result/{file_id}")
@validate_file_id
async def get_result(file_id: str):
    try:
        result_path = os.path.join(RESULT_DIR, f"{file_id}.png")
        if os.path.exists(result_path):
            return FileResponse(result_path)
        
        return JSONResponse(
            status_code=400,
            content={"message": "Result not found"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Invalid file ID: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Background Removal API")
    uvicorn.run(app, host="0.0.0.0", port=8080)