import cv2
import numpy as np
from os import path
from argparse import Namespace

MODEL_PATH = path.join(path.dirname(path.abspath(__file__)), "models")

args = Namespace(
    use_gpu=False,
    ir_optim=True,
    use_tensorrt=False,
    min_subgraph_size=15,
    det_algorithm="DB",
    det_model_dir=path.join(MODEL_PATH, "ppocrv4/det/det.onnx"),
    det_limit_side_len=960,
    det_limit_type="max",
    det_box_type="quad",
    det_db_thresh=0.3,
    det_db_box_thresh=0.6,
    det_db_unclip_ratio=1.5,
    max_batch_size=10,
    use_dilation=False,
    det_db_score_mode="fast",
    det_east_score_thresh=0.8,
    det_east_cover_thresh=0.1,
    det_east_nms_thresh=0.2,
    det_sast_score_thresh=0.5,
    det_sast_nms_thresh=0.2,
    det_pse_thresh=0,
    det_pse_box_thresh=0.85,
    det_pse_min_area=16,
    det_pse_scale=1,
    scales=[8, 16, 32],
    alpha=1.0,
    beta=1.0,
    fourier_degree=5,
    rec_algorithm="SVTR_LCNet",
    rec_model_dir=path.join(MODEL_PATH, "ppocrv4/rec/rec.onnx"),
    rec_image_inverse=True,
    rec_image_shape="3, 48, 320",
    rec_batch_num=6,
    max_text_length=25,
    rec_char_dict_path=path.join(MODEL_PATH, "ch_ppocr_server_v2.0/ppocr_keys_v1.txt"),
    use_space_char=True,
    drop_score=0.5,
    use_angle_cls=False,
    cls_model_dir=path.join(MODEL_PATH, "ppocrv4/cls/cls.onnx"),
    cls_image_shape="3, 48, 192",
    label_list=["0", "180"],
    cls_batch_num=6,
    cls_thresh=0.9,
    enable_mkldnn=False,
    cpu_threads=10,
    show_log=True,
)

def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def get_minarea_rect_crop(img, points):
    bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_a, index_b, index_c, index_d = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_a = 0
        index_d = 1
    else:
        index_a = 1
        index_d = 0
    if points[3][1] > points[2][1]:
        index_b = 2
        index_c = 3
    else:
        index_b = 3
        index_c = 2

    box = [points[index_a], points[index_b], points[index_c], points[index_d]]
    crop_img = get_rotate_crop_image(img, np.array(box))
    return crop_img
