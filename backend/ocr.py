from paddleocr import PaddleOCR
import numpy as np
from PIL.Image import Image
from typing import List
import logging
from threading import Lock

class OCRProcessor:
    def __init__(self, lang: str = 'ch'):
        logging.getLogger('ppocr').setLevel(logging.ERROR)
        self.ocr = PaddleOCR(use_angle_cls=False, lang=lang, enable_mkldnn=True)
        self.lock = Lock()

    def infer(self, image: Image):
        with self.lock:
            result = self.ocr.ocr(np.asarray(image), cls=True)
        if len(result) == 1 and result[0] is None:
            return []
        return [line[1][0] for line in result[0] if line[1][1] > 0.8]

    def batch_infer(self, images: List[Image]):
        return [self.infer(image) for image in images]

