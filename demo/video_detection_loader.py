from itertools import count
from threading import Thread
from queue import Queue

import cv2, sys
import numpy as np
from time import sleep
from tqdm import tqdm
import queue

import torch
import torch.multiprocessing as mp
from torchvision.transforms import functional as F
from pathlib import Path

from detector.apis import get_detector
from detector.Yolov5_DeepSort_Pytorch.yolov5.utils.datasets import LoadImages, LoadStreams
from detector.Yolov5_DeepSort_Pytorch.yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from detector.Yolov5_DeepSort_Pytorch.yolov5.utils.torch_utils import time_synchronized

class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image


class VideoDetectionLoader(object):
    '''
    This Class takes the video from the source (video file or camera) and tracks the person.
    '''
    def __init__(self, cfg, track_queue, action_queue, predictor_process):
        # cfg.detector = "tracker2"
        self.cfg = cfg
        self.detector = get_detector(cfg)
        self.input_path = cfg.input_path

        self.start_mill = cfg.start
        self.duration_mill = cfg.duration
        self.realtime = cfg.realtime

        stream = cv2.VideoCapture(self.input_path)
        assert stream.isOpened(), 'Cannot capture source'
        self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
        self.fps = stream.get(cv2.CAP_PROP_FPS)
        self.frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.videoinfo = {'fourcc': self.fourcc, 'fps': self.fps, 'frameSize': self.frameSize}
        stream.release()

        self._stopped = mp.Value('b', False)
        self.track_queue = track_queue
        self.action_queue = action_queue
        self.predictor_process = predictor_process

    def start_worker(self, target):
        p = mp.Process(target=target, args=())
        p.start()
        return p

    def start(self):
        # start a thread to pre process images for object detection
        if self.cfg.detector == 'tracker':
            self.image_preprocess_worker = self.start_worker(self.frame_preprocess)
        elif self.cfg.detector == 'tracker2':
            self.image_preprocess_worker = self.start_worker(self.yolo5_track)
        return self

    def stop(self):
        # end threads
        self.image_preprocess_worker.join()
        # clear queues
        self.clear_queues()

    def terminate(self):
        self._stopped.value = True
        self.stop()

    def clear_queues(self):
        self.clear(self.track_queue)
        self.clear(self.action_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def wait_and_put(self, queue, item):
        if not self.stopped:
            queue.put(item)

    def wait_and_get(self, queue):
        if not self.stopped:
            return queue.get()
    
    def wait_till_empty(self, queue):
        if not queue.empty():
            number_of_items = queue.qsize()
            print("{} item(s) to be processed".format(number_of_items))
            rest = number_of_items
            for i in tqdm(range(number_of_items)):
                if rest + i < number_of_items:
                    continue
                else:
                    rest = queue.qsize()
                    sleep(0.1)

            while True:
                if queue.empty():
                    print("Process completed")
                    return
                else:
                    sleep(0.1)

    def yolo5_track(self):
        dataset = LoadImages(self.input_path, img_size=self.detector.imgsz)

        cur_millis = 0

        for frame_idx, (path, img, im0s, stream) in enumerate(dataset):

            last_millis = cur_millis
            cur_millis = stream.get(cv2.CAP_PROP_POS_MSEC)

            if not self.realtime and self.duration_mill != -1 and cur_millis > self.start_mill + self.duration_mill:
                break

            img = torch.from_numpy(img).to(self.detector.device)
            
            img = img.half() if self.detector.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            # t1 = time_synchronized()
            pred = self.detector.model(img, augment=self.detector.augment)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred, self.detector.cfg.conf_thres, self.detector.cfg.iou_thres, classes=self.detector.cfg.classes, agnostic=self.detector.cfg.agnostic_nms)
            # t2 = time_synchronized()

            # Process detections
            img_det = None

            for i, det in enumerate(pred):  # detections per image
                if self.detector.webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                s += '%gx%g ' % img.shape[2:]  # print string
                # save_path = str(Path(out) / Path(p).name)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.detector.names[int(c)])  # add to string
                        
                    xywhs = xyxy2xywh(det[:, 0:4]) # det[:, 0:4] => [x1, y1, x2, y2]
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to deepsort
                    outputs = self.detector.deepsort.update(xywhs.cpu(), confs.cpu(), clss, im0)

                    orig_img = im0s[:, :, ::-1]
                    bboxes = []
                    scores = []
                    ids = []

                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)): 
                            bboxes.append(output[0:4])
                            scores.append(conf)
                            ids.append(output[4])

                    img_det = (orig_img, bboxes, scores, ids)
                        
                else:
                    self.detector.deepsort.increment_ages()
                    img_det = (orig_img, None, None, None)

                self.image_postprocess(img_det, (im0s, cur_millis))
                    
        self.wait_and_put(self.track_queue, (None, None, None, None))
        self.wait_and_put(self.action_queue, ("Done", self.videoinfo["frameSize"]))
        #self.wait_till_empty(self.action_queue)
        #self.wait_till_empty(self.track_queue)

        # This process needs to be finished after the predictor process
        # Otherwise, it will cause FileNotFoundError, if predictor is
        # overwhelmed
        predictor_process = self.predictor_process.value
        if predictor_process != -1:
            tqdm.write("Tracking finished. Showing feature extraction progress bar [ ready length / total length ](in msec).")
            initial = predictor_process - self.start_mill
            pbar = tqdm(total=int(last_millis)-self.start_mill, initial=initial, desc="Feature Extraction")
            last_pos = initial
            while predictor_process != -1:
                pbar.update(predictor_process-last_pos)
                if self.stopped:
                    break
                last_pos = predictor_process
                # try not to read shared value too frequent
                sleep(0.1)
                predictor_process = self.predictor_process.value
            pbar.update(int(last_millis)-last_pos)
            pbar.close()

        stream.release()

    def frame_preprocess(self):
        stream = cv2.VideoCapture(self.input_path)
        assert stream.isOpened(), 'Cannot capture source'
        self.nframes = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video total {self.nframes} frames loaded.")

        if not self.realtime:
            stream.set(cv2.CAP_PROP_POS_MSEC, self.start_mill)

        cur_millis = 0

        # keep looping infinitely
        for i in count():
            if self.stopped or (self.realtime and self.predictor_process.value == -1):
                stream.release()
                print("Video detection loader stopped")
                return
            if not self.track_queue.full():
                # otherwise, ensure the queue has room in it
                # The frame is in BGR format
                (grabbed, frame) = stream.read()
                last_millis = cur_millis
                cur_millis = stream.get(cv2.CAP_PROP_POS_MSEC)

                if not self.realtime and self.duration_mill != -1 and cur_millis > self.start_mill + self.duration_mill:
                    grabbed = False

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.wait_and_put(self.track_queue, (None, None, None, None, None))
                    self.wait_and_put(self.action_queue, ("Done", self.videoinfo["frameSize"]))
                    #self.wait_till_empty(self.action_queue)
                    #self.wait_till_empty(self.track_queue)

                    # This process needs to be finished after the predictor process
                    # Otherwise, it will cause FileNotFoundError, if predictor is
                    # overwhelmed
                    predictor_process = self.predictor_process.value
                    if predictor_process != -1:
                        tqdm.write("Tracking finished. Showing feature extraction progress bar [ ready length / total length ](in msec).")
                        initial = predictor_process - self.start_mill
                        pbar = tqdm(total=int(last_millis)-self.start_mill, initial=initial, desc="Feature Extraction")
                        last_pos = initial
                        while predictor_process != -1:
                            pbar.update(predictor_process-last_pos)
                            if self.stopped:
                                break
                            last_pos = predictor_process
                            # try not to read shared value too frequent
                            sleep(0.1)
                            predictor_process = self.predictor_process.value
                        pbar.update(int(last_millis)-last_pos)
                        pbar.close()

                    stream.release()
                    return

                # expected frame shape like (1,3,h,w) or (3,h,w)
                img_k = self.detector.image_preprocess(frame)

                if isinstance(img_k, np.ndarray):
                    img_k = torch.from_numpy(img_k)
                # add one dimension at the front for batch if image shape (3,h,w)
                if img_k.dim() == 3:
                    img_k = img_k.unsqueeze(0)

                im_dim_list_k = frame.shape[1], frame.shape[0]

                orig_img = frame[:, :, ::-1]
                im_name = str(i) + '.jpg'

                with torch.no_grad():
                    # Record original image resolution
                    im_dim_list_k = torch.FloatTensor(im_dim_list_k).repeat(1, 2)
                img_det = self.image_detection((img_k, orig_img, im_name, im_dim_list_k))
                self.image_postprocess(img_det, (frame, cur_millis))

    def image_detection(self, inputs):
        img, orig_img, im_name, im_dim_list = inputs
        if img is None or self.stopped:
            return (None, None, None, None)

        with torch.no_grad():
            dets = self.detector.images_detection(img, im_dim_list)
            if isinstance(dets, int) or dets.shape[0] == 0:
                return (orig_img, None, None, None)
            if isinstance(dets, np.ndarray):
                dets = torch.from_numpy(dets)
            dets = dets.cpu()
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            ids = dets[:, 6:7]

        boxes_k = boxes[dets[:, 0] == 0]
        if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
            return (orig_img, None, None, None)

        return (orig_img, boxes_k, scores[dets[:, 0] == 0], ids[dets[:, 0] == 0])

    def image_postprocess(self, inputs, extra):
        with torch.no_grad():
            (orig_img, boxes, scores, ids) = inputs
            if orig_img is None or self.stopped:
                self.wait_and_put(self.track_queue, (None, None, None, None))
                return

            # all parameters to be used in ava
            frame, cur_millis = extra
            input = (frame, cur_millis, boxes, scores, ids)

            # Passing these information to AVAPredictorWorker
            self.wait_and_put(self.action_queue, (input, self.videoinfo["frameSize"]))

            # Only return the tracking results to main thread
            self.wait_and_put(self.track_queue, (orig_img, boxes, scores, ids))

    def read_track(self):
        return self.wait_and_get(self.track_queue)

    def read_action(self):
        return self.wait_and_get(self.action_queue)

    @property
    def stopped(self):
        return self._stopped.value

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]
