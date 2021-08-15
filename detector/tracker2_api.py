import sys

from Yolov5_DeepSort_Pytorch.yolov5.utils.google_utils import attempt_download
from Yolov5_DeepSort_Pytorch.yolov5.models.experimental import attempt_load
from Yolov5_DeepSort_Pytorch.yolov5.utils.datasets import LoadImages, LoadStreams
from Yolov5_DeepSort_Pytorch.yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from Yolov5_DeepSort_Pytorch.yolov5.utils.torch_utils import select_device, time_synchronized
from Yolov5_DeepSort_Pytorch.yolov5.utils.plots import plot_one_box
from Yolov5_DeepSort_Pytorch.deep_sort_pytorch.utils.parser import get_config
from Yolov5_DeepSort_Pytorch.deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from detector.apis import BaseDetector
from tracker.preprocess import prep_image, prep_frame


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


class Tracker(BaseDetector):
    def __init__(self, cfg, opt):
        # cfg: tracker2_cfg.py에서 정의한 config
        # opt: demo.py에서 입력받은 args
        self.opt = opt
        self.cfg = cfg
        self.yolo_weights = cfg.YOLO_WEIGHTS
        self.deep_sort_weights = cfg.DEEP_WEIGHTS
        self.imgsz = cfg.IMG_SIZE
        self.evaluate = False
        self.augment = False
        self.webcam = False

        # initialize deepsort
        # self.orig_cfg: YOLOv5+DeepSORT 공식 코드에서 사용하는 config
        self.orig_cfg = get_config()
        self.orig_cfg.merge_from_file(cfg.CONFIG)
        attempt_download(self.deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
        self.deepsort = DeepSort(self.orig_cfg.DEEPSORT.REID_CKPT,
                            max_dist=self.orig_cfg.DEEPSORT.MAX_DIST, min_confidence=self.orig_cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=self.orig_cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=self.orig_cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=self.orig_cfg.DEEPSORT.MAX_AGE, n_init=self.orig_cfg.DEEPSORT.N_INIT, nn_budget=self.orig_cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # Initialize
        self.device = select_device(self.opt.device)
        # print(f"\n\nFile tracker2_api.py, in Tracker, self.device: {self.device}\n")
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = None
        self.load_model()

        # Load model
    def load_model(self):
        self.model = attempt_load(self.yolo_weights, map_location=self.opt.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.imgsz, s=stride)  # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        print("YOLOv5+DeepSORT Network successfully loaded")


    def image_preprocess(self, img_source):
        pass

    def images_detection(self, imgs, orig_dim_list):
        pass

    def detect_one_img(self, img_name):
        pass