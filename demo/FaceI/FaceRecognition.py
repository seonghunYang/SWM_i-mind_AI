import cv2, time, torch, os, traceback, pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from annoy import AnnoyIndex
from retinaface import RetinaFace
from preprocess import alignImage, preprocessImage, cropFullFace
from inference import imgToEmbedding, identifyFace
from visualization import drawFrameWithBbox
from backbones import get_model
from utils.utils import checkImgExtension 

def loadModel(backbone_name, weight_path, fp16=False):
    model = get_model(backbone_name, fp16=fp16)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    model = model.cuda()
    return model

def faceRecognition(input_video_path, out_video_path, annoy_tree, id_to_label, is_align=True):
    
    cap = cv2.VideoCapture(input_video_path)

    codec = cv2.VideoWriter_fourcc(*'XVID')

    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    vid_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    vid_writer = cv2.VideoWriter(out_video_path, codec, vid_fps, vid_size)

    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("총 Frame 갯수: ", frame_cnt)
    btime = time.time()

    detect_model = RetinaFace.build_model()
    recognition_model = loadModel("r50", "/content/drive/MyDrive/face_recognition_modules/glint360k_cosface_r50_fp16_0.1/backbone.pth")
    
    frame_idx = 0
    while True:
        has_frame, img_frame = cap.read()
        if not has_frame:
            print("처리 완료")
            break
        stime = time.time()
        try:
            detect_faces = RetinaFace.detect_faces(img_path=img_frame, model=detect_model)
            if (type(detect_faces) == dict):
                if is_align:
                    crop_face_imgs = alignImage(img_frame, detect_faces)
                else:
                    crop_face_imgs = []
                    for key in detect_faces.keys():
                        face = detect_faces[key]
                        facial_area = face['facial_area']
                        crop_face = cropFullFace(img_frame, facial_area)
                        crop_face = cv2.resize(crop_face, (112, 112))
                        crop_face_imgs.append(crop_face)
                identities = []
                for face_img in crop_face_imgs:
                    process_face_img, process_flip_face_img = preprocessImage(face_img)
                    embedding = imgToEmbedding(process_face_img, recognition_model, img_flip=process_flip_face_img)

                    annoy_idx, distacne = annoy_tree.get_nns_by_vector(embedding, 1, include_distances=True)
                    
                    if distacne[0] < 1:
                        identity = id_to_label[annoy_idx[0]]
                    else:
                        identity = ""

                    # identity = identity + str(round(distacne[0], 2))

                    identities.append(identity)
                img_frame = drawFrameWithBbox(img_frame, detect_faces, identities)
        except:
            print("에러가 발생했습니다. 현재까지 상황을 저장합니다")
            traceback.print_exc()
            break
        print('frame별 detection 수행 시간:', round(time.time() - stime, 4),frame_idx)
        frame_idx += 1
        vid_writer.write(img_frame)
    vid_writer.release()
    cap.release()

    print("최종 완료 수행 시간: ", round(time.time() - btime, 4))

def createEmbeddingDB(db_folder_path, db_save_path=None, is_align=True, img_show=False, build_tree=10):
    db = {
        "labels": [],
        "embedding": []
    }

    recognition_model = loadModel("r50", "/content/drive/MyDrive/face_recognition_modules/glint360k_cosface_r50_fp16_0.1/backbone.pth")

    face_folder_list = os.listdir(db_folder_path)
    for face_folder_name in face_folder_list:
        label = face_folder_name
        labels = []

        face_folder_path = db_folder_path + "/" + face_folder_name
        img_name_list = os.listdir(face_folder_path)
        for img_name in img_name_list:
            if not checkImgExtension(img_name):
                continue
            img = cv2.imread(face_folder_path + "/" + img_name)
            detect_faces = RetinaFace.detect_faces(img_path=img)
            if (type(detect_faces) == dict):
                if is_align:
                    crop_face_imgs = alignImage(img, detect_faces)
                else:
                    crop_face_imgs = []
                    for key in detect_faces.keys():
                        face = detect_faces[key]
                        facial_area = face['facial_area']
                        crop_face = cropFullFace(img, facial_area)
                        crop_face = cv2.resize(crop_face, (112, 112))
                        crop_face_imgs.append(crop_face)
            # embedding
                for face_img in crop_face_imgs:
                    process_face_img, process_flip_face_img = preprocessImage(face_img)
                    embedding = imgToEmbedding(process_face_img, recognition_model, img_flip=process_flip_face_img)
                    db["embedding"].append(embedding)
                    db['labels'].append(label)
                    labels.append(label)
                    if img_show:
                        plt.imshow(face_img)
                        plt.show()
    db['embedding'] = normalize(db['embedding'])

    #annoy 객체에 담기
    id_to_label = {}
    annoy_tree = AnnoyIndex(512, "euclidean")
    for idx in range(len(db['labels'])):
        annoy_tree.add_item(idx, db['embedding'][idx])
        id_to_label[idx] = db['labels'][idx]
        
    annoy_tree.build(build_tree)
    if db_save_path:
        annoy_tree.save(db_save_path+"_db.ann")
        with open(db_save_path +"_id_to_label.pickle", "wb") as f:
            pickle.dump(id_to_label, f)

    return annoy_tree, id_to_label

def trackingIdToFaceID(images_by_id, final_fuse_id, db_folder_path):
    track_id = list(final_fuse_id.keys())
    track_id_to_face_id = dict()

    #모델 로드
    detect_model = RetinaFace.build_model()
    recognition_model = loadModel("r50", "/content/drive/MyDrive/face_recognition_modules/glint360k_cosface_r50_fp16_0.1/backbone.pth")

    #db 로드
    annoy_tree = AnnoyIndex(512, "euclidean")
    annoy_tree.load(db_save_folder+"db.ann")
    with open(db_save_folder +"db.pickle", "rb") as f:
        id_to_label = pickle.load(f)

    for id in track_id:
        print("{} 시작 개수 :{}".format(id, len(images_by_id[id])))
        frequency_face_id = {
            "unknown": 0
        }
        for idx, img in enumerate(images_by_id[id]):
            stime = time.time()
            try:
                detect_faces = RetinaFace.detect_faces(img_path=img, model=detect_model)
            except:
                continue
            if (type(detect_faces) == dict):
                crop_face_imgs = []
                for key in detect_faces.keys():
                        face = detect_faces[key]
                        facial_area = face['facial_area']
                        crop_face = cropFullFace(img, facial_area)
                        crop_face = cv2.resize(crop_face, (112, 112))
                        crop_face_imgs.append(crop_face)

                for face_img in crop_face_imgs:
                    process_face_img, process_flip_face_img = preprocessImage(face_img)
                    embedding = imgToEmbedding(process_face_img, model, img_flip=process_flip_face_img)
                    annoy_idx, distance = annoy_tree.get_nns_by_vector(embedding, 1, include_distances=True)
                    face_id = id_to_label[annoy_idx[0]]
                    if distance[0] < 1.3:
                        if frequency_face_id.get(face_id):
                            frequency_face_id[face_id] += 1
                        else:
                            frequency_face_id[face_id] = 1
                    else:
                        frequency_face_id['unknown'] += 1
            print(time.time() -stime, idx)
        mode_face_id = max(frequency_face_id, key=frequency_face_id.get)
        track_id_to_face_id[id] = mode_face_id

    return track_id_to_face_id