from .predict_system import TextSystem
from .utils import args as default_args
import numpy as np
from PIL.Image import Image
from os import path
from typing import List
from threading import Lock

class OCRProcessor:
    def __init__(self, lang: str = 'ch', model_base_path: str = ''):
        det_model_dir=path.join(model_base_path, "ppocrv4/det/det.onnx")
        rec_model_dir=path.join(model_base_path, "ppocrv4/rec/rec.onnx")
        cls_model_dir=path.join(model_base_path, "ppocrv4/cls/cls.onnx")
        rec_char_dict_path=path.join(model_base_path, "ch_ppocr_server_v2.0/ppocr_keys_v1.txt")
        self.ocr = ONNXPaddleOcr(
            use_angle_cls=False,
            lang=lang,
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            cls_model_dir=cls_model_dir,
            rec_char_dict_path=rec_char_dict_path,
        )
        self.lock = Lock()

    def infer(self, image: Image):
        with self.lock:
            result = self.ocr.ocr(np.asarray(image), cls=False)
        if len(result) == 1 and result[0] is None:
            return []
        return [line[1][0] for line in result[0] if line[1][1] > 0.8]

    def batch_infer(self, images: List[Image]):
        return [self.infer(image) for image in images]


class ONNXPaddleOcr(TextSystem):
    def __init__(self, **kwargs):
        params = default_args
        params.__dict__.update(**kwargs)
        super().__init__(params)

    def ocr(self, img, det=True, rec=True, cls=False):
        if cls == True and self.use_angle_cls == False:
            print(
                "Since the angle classifier is not initialized, the angle classifier will not be uesd during the forward process"
            )

        if det and rec:
            ocr_res = []
            dt_boxes, rec_res = self.__call__(img, cls)
            tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
            ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            ocr_res = []
            dt_boxes = self.text_detector(img)
            tmp_res = [box.tolist() for box in dt_boxes]
            ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []

            if not isinstance(img, list):
                img = [img]
            if self.use_angle_cls and cls:
                img, cls_res_tmp = self.text_classifier(img)
                if not rec:
                    cls_res.append(cls_res_tmp)
            rec_res = self.text_recognizer(img)
            ocr_res.append(rec_res)

            if not rec:
                return cls_res
            return ocr_res