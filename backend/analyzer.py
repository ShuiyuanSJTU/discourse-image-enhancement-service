import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from PIL import Image
from typing import List, Dict, Tuple
from .model import ImageInfo, ProcessingImage, AnalysisResult
from .image_preprocessor import ImagePreprocessor
from .ocr import OCRProcessor
from .description import DescriptionPipeline
from .translation import TranslationLocalPipeline, TranslationLLMAgent
from .config import Config

class ImageAnalyzer:
    @staticmethod
    def preprocess_images(image_info: List[ImageInfo], config: Config) -> Dict[str, ProcessingImage]:
        preprocessor = ImagePreprocessor(image_info, config = config)
        return preprocessor.preprocess()

    @staticmethod
    def extract_downloaded_images(processing_images: Dict[str, ProcessingImage]) -> Tuple[List[Image.Image], List[str]]:
        images = []
        sha1 = []
        for processing_image in processing_images.values():
            if processing_image.image is not None:
                images.append(processing_image.image)
                sha1.append(processing_image.sha1)
        assert len(images) == len(sha1)
        return images, sha1
    
    def __init__(self, config: Config):
        self.config = config
        if config.ocr_enabled:
            self.ocr_processor = OCRProcessor(
                lang=config.ocr_lang
            )
        if config.description_enabled:
            self.description_pipeline = DescriptionPipeline(
                model=config.description_model
            )
        if config.translation_enabled:
            if config.llm_translation_enabled:
                self.translation_pipeline = TranslationLLMAgent(
                    model=config.llm_translation_model,
                    endpoint=config.llm_translation_endpoint,
                    api_key=config.llm_translation_api_key,
                    temperature=config.llm_translation_temperature,
                    max_tokens=config.llm_translation_max_tokens,
                    prompt=config.llm_translation_prompt
                )
            elif config.local_translation_enabled:
                self.translation_pipeline = TranslationLocalPipeline(
                    model=config.local_translation_model
                )
            else:
                self.translation_pipeline = None

    def analyze_images(self, image_info: List[ImageInfo], lang: str = None) \
            -> List[AnalysisResult]:
        processing_images = self.preprocess_images(image_info, config=self.config)
        images, sha1_list = self.extract_downloaded_images(processing_images)
        if self.ocr_processor is not None:
            ocr_results = self.ocr_processor.batch_infer(images)
            for sha1, ocr_result in zip(sha1_list, ocr_results):
                processing_images[sha1].result.ocr_result = ocr_result

        if self.description_pipeline is not None:
            description_results = self.description_pipeline.batch_infer(images)
            if self.translation_pipeline is not None:
                description_results = self.translation_pipeline.batch_infer(description_results)
            for sha1, description in zip(sha1_list, description_results):
                processing_images[sha1].result.description = description
        
        return [processing_image.result for processing_image in processing_images.values()]
