import requests
from PIL import Image
from .model import ImageInfo, ProcessingImage, AnalysisResult
from .config import Config
from typing import List, Dict
import hashlib
from io import BytesIO
from fnmatch import fnmatch

class ImagePreprocessor:
    def __init__(self, image_info: List[ImageInfo], config: Config):
        self.image_info = image_info
        self.config = config
        self.processing_images = {}

        for image in image_info:
            self.processing_images[image.sha1] = ProcessingImage(sha1=image.sha1, info=image, result=AnalysisResult(sha1=image.sha1))

    def ensure_whitelist(self, url: str):
        if not self.config.download_whitelist_enabled:
            return
        matched = any(fnmatch(url, pattern) for pattern in self.config.download_allowed_urls)
        if not matched:
            raise ValueError(f"URL {url} not in download whitelist")

    def download_image(self, 
                    check_sha1: bool = False):
        for processing_image in self.processing_images.values():
            image = processing_image.info
            try:
                self.ensure_whitelist(image.url)
                response = requests.get(image.url)
                if response.status_code != 200:
                    raise ValueError(f"Failed to download image {image.url}, status code {response.status_code}")
                if check_sha1 and image.sha1 is not None:
                    sha1 = hashlib.sha1(response.content).hexdigest().lower()
                    if sha1.lower() != image.sha1.lower():
                        raise ValueError(f"SHA1 mismatch for image {image.url}")
                image_data = Image.open(BytesIO(response.content)).convert('RGB')
                processing_image.image = image_data
            except Exception as e:
                processing_image.result.success = False
                processing_image.result.error = str(e)
    
    def resize_image(self, 
                    max_size: int = 2048):
        for processing_image in self.processing_images.values():
            image = processing_image.image
            if image is not None:
                width, height = image.size
                if width > max_size or height > max_size:
                    ratio = min(max_size / width, max_size / height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    processing_image.image = image.resize((new_width, new_height))

    def preprocess(self) -> Dict[str, ProcessingImage]:
        self.download_image(self.config.download_check_sha1)
        self.resize_image(self.config.download_max_size)
        return self.processing_images
