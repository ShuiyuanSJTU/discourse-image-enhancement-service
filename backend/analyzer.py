from PIL import Image
from typing import List, Dict, Tuple
from .model import ImageInfo, ProcessingImage, AnalysisResult
from .image_preprocessor import ImagePreprocessor
from .ocr import OCRProcessor
from .embedding import EmbeddingProcessor
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
                lang=config.ocr_lang,
                model_base_path=config.ocr_onnx_model_path
            )
        if config.clip_enabled:
            self.clip_processor = EmbeddingProcessor(
                model_base_path=config.clip_onnx_model_path
            )

    def analyze_images(self, image_info: List[ImageInfo], lang: str = None,
                        analyze_ocr: bool = True, analyze_embedding: bool = True) \
            -> List[AnalysisResult]:
        processing_images = self.preprocess_images(image_info, config=self.config)
        images, sha1_list = self.extract_downloaded_images(processing_images)
        if analyze_ocr and self.ocr_processor is not None:
            ocr_results = self.ocr_processor.batch_infer(images)
            for sha1, ocr_result in zip(sha1_list, ocr_results):
                processing_images[sha1].result.ocr_result = ocr_result

        if analyze_embedding and self.clip_processor is not None:
            embedding_results = self.clip_processor.batch_infer(images)
            for sha1, embedding in zip(sha1_list, embedding_results):
                processing_images[sha1].result.embedding = embedding.tolist()

        return [processing_image.result for processing_image in processing_images.values()]
