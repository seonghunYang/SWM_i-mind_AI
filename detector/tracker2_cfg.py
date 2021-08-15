from easydict import EasyDict as edict

cfg = edict()
cfg.CONFIG = '../detector/Yolov5_DeepSort_Pytorch/deep_sort_pytorch/configs/deep_sort.yaml'
cfg.YOLO_WEIGHTS = '../detector/Yolov5_DeepSort_Pytorch/yolov5/weights/crowdhuman_yolov5m.pt'
cfg.DEEP_WEIGHTS = '../detector/Yolov5_DeepSort_Pytorch/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
cfg.IMG_SIZE =  640
cfg.iou_thres =  0.5
cfg.conf_thres = 0.4
cfg.classes = 0
cfg.agnostic_nms = False