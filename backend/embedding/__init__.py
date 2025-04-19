import onnxruntime
import numpy as np
from os import path
from PIL.Image import Image
from typing import List, Union
from .image import CLIPImageEmbedding
from .text import CLIPTextEmbedding

onnxruntime.set_default_logger_severity(3)

class EmbeddingProcessor:
    def __init__(self, model_base_path: str = ""):
        self.image_model = CLIPImageEmbedding(path.join(model_base_path, "clip/vit-b-16.img.fp16.onnx"))
        self.text_model = CLIPTextEmbedding(path.join(model_base_path, "clip/vit-b-16.txt.fp16.onnx"))

    def infer(self, input: Union[Image, str]) -> np.ndarray:
        if isinstance(input, Image):
            return self.image_model(input)[0]
        elif isinstance(input, str):
            return self.text_model(input)[0]
    
    def batch_infer(self, inputs: List[Union[Image,str]]) -> List[np.ndarray]:
        return [self.infer(input) for input in inputs]
