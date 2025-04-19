import onnxruntime
import numpy as np
from .tokenizer import tokenize

class CLIPTextEmbedding:
    def __init__(self, model_path: str = "vit-b-16.txt.fp16.onnx"):
        txt_sess_options = onnxruntime.SessionOptions()
        self.txt_session = onnxruntime.InferenceSession(model_path,
                                                sess_options=txt_sess_options,
                                                providers=["CPUExecutionProvider"])

    def __call__(self, text: str):
        text_feature = self.txt_session.run(["unnorm_text_features"], {"text":tokenize([text])})[0]
        text_feature /= np.linalg.norm(text_feature, axis=-1, keepdims=True)
        return text_feature