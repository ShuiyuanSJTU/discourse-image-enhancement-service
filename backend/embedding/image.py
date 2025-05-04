import onnxruntime
import numpy as np
from PIL.Image import Image


class CLIPImageEmbedding:
    def __init__(self, model_path: str = "vit-b-16.img.fp16.onnx"):
        img_sess_options = onnxruntime.SessionOptions()
        self.img_session = onnxruntime.InferenceSession(model_path,
                                                sess_options=img_sess_options,
                                                providers=["CPUExecutionProvider"])
        
    @staticmethod
    def preprocess(image: Image, image_size: int = 224) -> np.ndarray:
        image = image.convert("RGB").resize((image_size, image_size))
        image = np.array(image).astype(np.float32) / 255.0
        # Normalize (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        image = (image - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array([0.26862954, 0.26130258, 0.27577711])
        image = np.transpose(image, (2, 0, 1))
        return image.astype(np.float32)[None, :]

    def __call__(self, image: Image):
        image = self.preprocess(image)
        image_features = self.img_session.run(["unnorm_image_features"], 
                                              {"image": image})[0]
        return image_features
