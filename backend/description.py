from transformers import pipeline
import torch
from PIL.Image import Image
from typing import List

class DescriptionPipeline:
    def __init__(self, model: str = "Salesforce/blip-image-captioning-large"):
        torch.set_num_threads(8)
        self.model = pipeline("image-to-text", model=model)

    def infer(self, image: Image):
        return self.model(image)[0]['generated_text']
    
    def batch_infer(self, image: List[Image]):
        return [result[0]['generated_text'] for result in self.model(image)]