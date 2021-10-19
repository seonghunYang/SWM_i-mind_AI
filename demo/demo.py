# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by DrMaemi (leesh1510@ajou.ac.kr)
# -----------------------------------------------------
import sys
# print(f"\n\nsys.path: {sys.path}\n\n")
sys.path.insert(0, '../detector/Yolov5_DeepSort_Pytorch')
sys.path.insert(0, '../detector/Yolov5_DeepSort_Pytorch/yolov5')
# print(f"\n\nsys.path: {sys.path}\n\n")

import argparse
from time import sleep
from itertools import count
from tqdm import tqdm

import numpy as np
import torch
from reid import REID
import operator

from visualizer import AVAVisualizer, EmotionVisualizer
from action_predictor import AVAPredictorWorker

#pytorch issuse #973
import resource
from annoy import AnnoyIndex
import time
from torchreid.data.transforms import build_transforms
from PIL import Image
import json
from catch_aws_client import AWSClient
from catch_logger import EmotionLogger

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))
import cProfile

def main():
    parser = argparse.ArgumentParser(description='Action Detection Demo')
    parser.add_argument(
        "--webcam",
        dest="webcam",
        help="Use webcam as input",
        action="store_true",
    )
    parser.add_argument(
        "--video-path",
        default="input.mp4",
        help="The path to the input video",
        type=str,
    )
    parser.add_argument(
        "--output-path",
        default="output.mp4",
        help="The path to the video output",
        type=str,
    )
    parser.add_argument(
        "--cpu",
        dest="cpu",
        help="Use cpu",
        action="store_true",
    )
    parser.add_argument(
        "--cfg-path",
        default="../config_files/resnet101_8x8f_denseserial.yaml",
        help="The path to the cfg file",
        type=str,
    )
    parser.add_argument(
        "--weight-path",
        default="../data/models/aia_models/resnet101_8x8f_denseserial.pth",
        help="The path to the model weights",
        type=str,
    )
    parser.add_argument(
        "--visual-threshold",
        default=0.5,
        help="The threshold of visualizer",
        type=float,
    )
    parser.add_argument(
        "--show-id",
        default=False,
        help="visualize track ids",
        action="store_true",
    )
    parser.add_argument(
        "--start",
        default=0,
        help="Start reading video at which millisecond",
        type=int,
    )
    parser.add_argument(
        "--duration",
        default=-1,
        help="The duration of detection",
        type=int,
    )
    parser.add_argument(
        "--detect-rate",
        default=4,
        help="Rate(fps) to update action labels",
        type=int
    )
    parser.add_argument(
        "--common-cate",
        default=False,
        help="Using common category model",
        action="store_true"
    )
    parser.add_argument(
        "--hide-time",
        default=False,
        help="Not show the timestamp at the corner",
        action="store_true"
    )
    parser.add_argument(
        "--tracker-box-thres",
        default=0.1,
        help="The box threshold for tracker",
        type=float,
    )
    parser.add_argument(
        "--tracker-nms-thres",
        default=0.4,
        help="The nms threshold for tracker",
        type=float,
    )
    parser.add_argument(
        "--reid",
        default=False,
        help="do re-identification",
        action="store_true",
    )
    parser.add_argument(
        "--s3",
        default=False,
        help="analyze s3 video",
        action="store_true",
    )
    parser.add_argument(
        "--bucket-name",
        default='drmaemi.com',
        help="s3 bucket name to create aws client object",
        type=str,
    )
    parser.add_argument(
        "--s3-file-key",
        default='public/test/double_single.mp4',
        help="analyze s3 video",
        type=str
    )
    parser.add_argument(
        "--final-output-path",
        default="final_output.mp4",
        help="The path to the video final output",
        type=str,
    )

    args = parser.parse_args()

    args.input_path = 0 if args.webcam else args.video_path
    args.device = torch.device("cpu" if args.cpu else "cuda")
    args.realtime = True if args.webcam else False

    # Configuration for Tracker. Currently Multi-gpu is not supported
    args.gpus = "0"
    args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
    args.min_box_area = 0
    args.tracking = True
    args.detector = "tracker2"
    args.debug = False

    reid = REID()
    print("ReID model loaded")

    aws_client = None

    if args.webcam:
        print('Starting webcam demo, press Ctrl + C to terminate...')
    elif args.s3:
        aws_client = AWSClient(args.bucket_name)
        save_path = 'downloaded.mp4'
        print('Downloading video file ...')
        aws_client.download_file(args.s3_file_key, save_path)
        print(f'Downloaded: {save_path}')
        args.input_path = args.video_path = save_path
        print('Call rekognition api...')
        aws_client.start_face_detection(args.s3_file_key)
        aws_client.get_face_detection()
        print('Called.')
    else:
        print('Starting video demo, video path: {}'.format(args.video_path))

    fuse_queue = torch.multiprocessing.Queue()
    # Initialise Visualizer
    video_writer = AVAVisualizer(
        fuse_queue,
        args.input_path,
        args.output_path,
        args.realtime,
        args.start,
        args.duration,
        (not args.hide_time),
        confidence_threshold = args.visual_threshold,
        common_cate = args.common_cate,
        show_id = args.show_id,
        detector = args.detector
    )

    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    ava_predictor_worker = AVAPredictorWorker(args)
    pred_done_flag = False

    box_imgs_by_id = {}
    id_with_box_imgs = []
    ids_per_frame = []
    ann = AnnoyIndex(2048, 'euclidean')
    ann_idxs_by_id = {}
    reid_features = []
    ann_idx = 0
    
    print("Showing tracking progress bar (in fps). Other processes are running in the background.")
    try:
        for i in tqdm(count(), desc="Tracker Progress", unit=" frame"):
            with torch.no_grad():
                (orig_img, boxes, scores, ids) = ava_predictor_worker.read_track()

                if orig_img is None:
                    if not args.realtime:
                        ava_predictor_worker.compute_prediction()
                    break

                if args.realtime:
                    if len(orig_img.shape) == 4:
                    # if args.detector == "tracker2":
                        orig_img = np.array(orig_img[0, :, :, :])
                    result = ava_predictor_worker.read()
                    flag = video_writer.realtime_write_frame(result, orig_img, boxes, scores, ids)
                    if not flag:
                        break
                else:
                    if args.reid:
                        try:
                            ids_per_frame.append(set(map(int, ids)))

                            for bbox, id in zip(boxes, map(int, ids)):
                                box_img = orig_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                                
                                if id not in box_imgs_by_id:
                                    box_imgs_by_id[id] = [box_img]
                                    ann_idxs_by_id[id] = []
                                else:
                                    box_imgs_by_id[id].append(box_img)

                                    if len(box_imgs_by_id[id]) == 10:
                                        reid_features = reid._features_opt(box_imgs_by_id[id])
                                        # reid_features.size() -> torch.Size([10, 2048])

                                        for feat in reid_features:
                                            ann.add_item(ann_idx, feat)
                                            ann_idxs_by_id[id].append(ann_idx)
                                            ann_idx += 1

                                        box_imgs_by_id[id] = []
                                
                        except TypeError:
                            pass

                    video_writer.send_track((boxes, ids))
                    while not pred_done_flag:
                        result = ava_predictor_worker.read()
                        if result is None:
                            break
                        elif result == "done":
                            pred_done_flag = True
                        else:
                            video_writer.send(result)
        
        for id, box_imgs in box_imgs_by_id.items():
            if box_imgs:
                reid_features = reid._features_opt(box_imgs_by_id[id])

                for feat in reid_features:
                    ann.add_item(ann_idx, feat)
                    ann_idxs_by_id[id].append(ann_idx)
                    ann_idx += 1
        
        box_imgs_by_id = {}
            
    except KeyboardInterrupt:
        print("Keyboard Interrupted")

    if not args.realtime:
        threshold = 320
        exist_ids = set()
        final_fuse_id = dict()
        if args.reid:
            ann_tree_build_start = time.time()
            ann_tree_size = 5
            ann.build(ann_tree_size)
            print(f"Annoy tree build took {round(time.time()-ann_tree_build_start, 3)} seconds")

            reid_start = time.time()
            for f in ids_per_frame:
                if f:
                    if len(exist_ids) == 0:
                        for i in f:
                            final_fuse_id[i] = [i]

                        exist_ids = exist_ids or f
                    else:
                        new_ids = f-exist_ids
                        for nid in new_ids:
                            dis = []
                            if len(ann_idxs_by_id[nid]) < 10:
                                exist_ids.add(nid)
                                continue

                            unpickable = []
                            for i in f: # f = ids
                                for key, item in final_fuse_id.items(): # {key: 병합된 id, value: 병합되기 전 id들 리스트}
                                    if i in item:
                                        unpickable += final_fuse_id[key]
                            print('exist_ids {} unpickable {}'.format(exist_ids, unpickable))

                            for oid in (exist_ids-set(unpickable))&set(final_fuse_id.keys()):
                                try:
                                    tmp = []
                                    for ann_idx_by_nid in ann_idxs_by_id[nid]:
                                        for ann_idx_by_oid in ann_idxs_by_id[oid]:
                                            tmp.append(ann.get_distance(ann_idx_by_nid, ann_idx_by_oid))
                                    
                                    # print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                                    tmp = np.min(tmp)
                                    dis.append([oid, tmp])

                                except KeyError:
                                    pass

                            exist_ids.add(nid)

                            if not dis:
                                final_fuse_id[nid] = [nid]
                                continue

                            dis.sort(key=operator.itemgetter(1))

                            if dis[0][1] < threshold:
                                combined_id = dis[0][0]
                                ann_idxs_by_id[combined_id] += ann_idxs_by_id[nid]
                                del ann_idxs_by_id[nid]
                                final_fuse_id[combined_id].append(nid)
                            else:
                                final_fuse_id[nid] = [nid]

            print('Final ids and their sub-ids:', final_fuse_id)
            print(f"Re-ID took {round(time.time()-reid_start, 3)} seconds")
            
        final_fuse_id_reverse = dict()
        
        if args.reid:
            for final_id, sub_ids in final_fuse_id.items():
                for sub_id in sub_ids:
                    final_fuse_id_reverse[sub_id] = final_id

        # for passing 'final_fuse_id_reverse' to video_writer
        fuse_queue.put(final_fuse_id_reverse)
        
        video_writer.send_track("DONE")
        while not pred_done_flag:
            result = ava_predictor_worker.read()

            if result is None:
                sleep(0.1)
            elif result == "done":
                pred_done_flag = True
            else:
                video_writer.send(result)

        video_writer.send("DONE")
        tqdm.write("Showing video writer progress (in fps).")
        video_writer.progress_bar(i)

    video_writer.close()
    ava_predictor_worker.terminate()

    if args.s3:
        all_rekog_results = aws_client.get_all_rekog_results()
        face_boxes_and_emotions_by_timestamp = aws_client.get_face_boxes_and_emotions(all_rekog_results)

        splited_by_path = args.s3_file_key.split('/')
        splited_by_bar = splited_by_path[-1].split('_')

        video_upload_key = '/'.join(splited_by_path[:-1])+'/'+'_'.join(splited_by_bar[:-1])+'_processed.mp4'
        action_upload_key = '/'.join(splited_by_path[:-1])+'/'+'_'.join(splited_by_bar[:-1])+'_action.log'
        emotion_upload_key = '/'.join(splited_by_path[:-1])+'/'+'_'.join(splited_by_bar[:-1])+'_emotion.log'

        emotion_logger = EmotionLogger('../logs').log(face_boxes_and_emotions_by_timestamp)
        emotion_visualizer = EmotionVisualizer(args.output_path)
        emotion_visualizer.output(args.final_output_path)

        print('Uploading final output video...')
        aws_client.upload_file(args.final_output_path, video_upload_key)
        print(f'Uploaded: [LOCAL]{args.final_output_path} → [S3 Bucket]{video_upload_key}')

        print('Uploading action log...')
        aws_client.upload_file('../logs/action.log', action_upload_key)
        print(f'Uploaded: [LOCAL]../logs/action.log → [S3 Bucket]{action_upload_key}')

        print('Uploading emotion log...')
        aws_client.upload_file('../logs/emotion.log', emotion_upload_key)
        print(f'Uploaded: [LOCAL]../logs/emotion.log → [S3 Bucket]{emotion_upload_key}')

if __name__ == "__main__":
    main()
