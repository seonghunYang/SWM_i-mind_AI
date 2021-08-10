import cv2, time, torch, os, traceback, logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from retinaface import RetinaFace
from preprocess import alignImage, preprocessImage, cropFullFace
from inference import imgToEmbedding, identifyFace, calculateDistance
from visualization import drawFrameWithBbox
from backbones import get_model
from utils.utils import checkImgExtension 
from FaceRecognition import loadModel



def collectFaceImageWithSeed(input_video_path, model_path, seed, threshold):
    
    cap = cv2.VideoCapture(input_video_path)

    codec = cv2.VideoWriter_fourcc(*'XVID')

    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("총 Frame 갯수: ", frame_cnt)
    btime = time.time()

    detect_model = RetinaFace.build_model()
    if type(model_path) == str:
        recognition_model = loadModel("r50", model_path)
    else:
        recognition_model = model_path
    
    collect_img = [[] for _ in range(len(seed['labels']))]
    
    frame_idx = 0
    while True:
        cap.set(1, frame_idx)
        has_frame, img_frame = cap.read()
        if not has_frame:
            print("처리 완료")
            break
        stime = time.time()
        try:
            detect_faces = RetinaFace.detect_faces(img_path=img_frame, model=detect_model)
        except:
            print('프레임 에러 발생 다음 프레임 이동')
            frame_idx += 3
            continue
        if (type(detect_faces) == dict):
            crop_face_imgs = []
            for key in detect_faces.keys():
                face = detect_faces[key]
                facial_area = face['facial_area']
                crop_face = cropFullFace(img_frame, facial_area)
                crop_face = cv2.resize(crop_face, (112, 112))
                crop_face_imgs.append(crop_face)

            for face_img in crop_face_imgs:
                process_face_img, process_flip_face_img = preprocessImage(face_img)
                embedding = imgToEmbedding(process_face_img, recognition_model, img_flip=process_flip_face_img)
                # 신원 확인이 아니라 비교 하고 이미지 수집해야함
                seed_idx_list, seed_distance_list = verifyWithSeed(embedding, seed, threshold)
                if len(seed_idx_list) > 0:
                    for idx, seed_idx in enumerate(seed_idx_list):
                        collect_img[seed_idx].append([face_img, seed_distance_list[idx]])
        print('frame별 detection 수행 시간:', round(time.time() - stime, 4),frame_idx)
        frame_idx += 3
    cap.release()

    print("최종 완료 수행 시간: ", round(time.time() - btime, 4))
    
    return collect_img

def verifyWithSeed(embedding, seed, threshold):
    idx_list = []
    distance_list = []
    for idx, seed_embedding in enumerate(seed['embedding']):
        distance = calculateDistance(embedding, seed_embedding)
        
        if distance < threshold:
            idx_list.append(idx)
            distance_list.append(distance)

    return idx_list, distance_list
    

def createEmbedingSeed(root_path, folder_name , model, img_show=False):
    seed = {
        "labels": [],
        "embedding": []
    }
    labels = []
    seed_folder_path = root_path + "/" + folder_name
    seed_img_list = os.listdir(seed_folder_path)
    for idx, seed_img in enumerate(seed_img_list):
        if not checkImgExtension(seed_img):
            continue
        img = cv2.imread(seed_folder_path + "/" + seed_img)
        detect_face = RetinaFace.detect_faces(img_path=img)
        if (type(detect_face) == dict):
            key = list(detect_face.keys())[0]
            face = detect_face[key]
            facial_area = face['facial_area']
            crop_face = cropFullFace(img, facial_area)
            crop_face = cv2.resize(crop_face, (112, 112))
            
            #embedding
            process_face_img, process_flip_face_img = preprocessImage(crop_face)
            embedding = imgToEmbedding(process_face_img, model, img_flip=process_flip_face_img)
            seed["embedding"].append(embedding)
            seed['labels'].append(idx)
            if img_show:
                plt.imshow(crop_face)
                plt.show()
    seed['embedding'] = normalize(seed['embedding'])
    return seed

def filterCosineSimilarity(collect_img, model):
    embeddings = []
    for i in range(len(collect_img)):
        img = collect_img[i][0]
        im, flip_im = preprocessImage(img)
        embedding = imgToEmbedding(im, model, img_flip=flip_im)
        embeddings.append(embedding)

    similarity_metrics = cosine_similarity(embeddings)

    duplicate_image_idx = set()

    for i in range(len(similarity_metrics[0])):
        if i in duplicate_image_idx:
            continue
        for j in range(i+1, len(similarity_metrics[1])):
            if similarity_metrics[i][j] >= 0.8:
                duplicate_image_idx.add(j)
    new_collect_img = []

    for i in range(len(collect_img)):
        if i in duplicate_image_idx:
            continue
        new_collect_img.append(collect_img[i])
    return new_collect_img

def selectNearImageAndSave(filter_collect_img, number):
    select_img = filter_collect_img.copy()
    if len(select_img) > 60:
      select_img.sort(key=lambda x: x[1])
      select_img = select_img[:60]
    for idx, img in enumerate(select_img):
      directory = "../our_children_changed/dataset/{}".format(number)
      if not os.path.exists(directory):
        os.makedirs(directory)
      plt.imsave(directory + "/{}.jpg".format(idx), img[0])


def collectDatasetPipeline(root_path, page, number, model):
    #seed
    logging.basicConfig(filename=root_path + "/log/page{}.log".format(page), format="%(asctime)s %(levelname)s %(message)s")
    try:
        seed = createEmbedingSeed(root_path + "/seed", str(number), model, img_show=True)
        print("시드 라벨 개수: ", len(seed['labels']))
        #filter-1
        collect_img = collectFaceImageWithSeed(root_path + "/video/page{}/{}.mp4".format(page, number), model, seed, 1)
        print("1차 필터링 완료")
        for i in range(len(collect_img)):
            print("{}번: {}개".format(i,len(collect_img[i])))
        #filter-1 and filter-1
        for i in range(len(collect_img)):
            filter_collect_img = filterCosineSimilarity(collect_img[i], model)
            print("{}번 2차필터링 완료: {}개".format(i ,len(filter_collect_img)) )
            #save
            save_folder_name = int(number) + int(i)
            selectNearImageAndSave(filter_collect_img, save_folder_name)
            print("{}번 3차 필터링 완료 저장 끝".format(i))
    except:
        logging.error(traceback.format_exc())
        traceback.print_exc()

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            os.makedirs(directory+ "-1")
    except OSError:
        print('Error: Creating directory: ' + directory)


def filterCosineSimilarityForImage(collect_img, model):
    embeddings = []
    for i in range(len(collect_img)):
        img = collect_img[i]
        im, flip_im = preprocessImage(img)
        embedding = imgToEmbedding(im, model, img_flip=flip_im)
        embeddings.append(embedding)

    similarity_metrics = cosine_similarity(embeddings)

    duplicate_image_idx = set()

    for i in range(len(similarity_metrics[0])):
        if i in duplicate_image_idx:
            continue
        for j in range(i+1, len(similarity_metrics[1])):
            if similarity_metrics[i][j] >= 0.8:
                duplicate_image_idx.add(j)
    new_collect_img = []

    for i in range(len(collect_img)):
        if i in duplicate_image_idx:
            continue
        new_collect_img.append(collect_img[i])
    return new_collect_img

def detectFaceAndFilterSimilarity(root_path):
    # 초기 설정
    actor_folder_list = os.listdir(root_path+"/raw_image")
    detect_model = RetinaFace.build_model()
    recognition_model = loadModel("r50", "/content/drive/MyDrive/face_recognition_modules/glint360k_cosface_r50_fp16_0.1/backbone.pth")
    
    for actor_folder in actor_folder_list:
        print(actor_folder, "처리 시작")
        stime = time.time()

        save_folder_path = root_path + "/face/{}".format(actor_folder)
        if os.path.exists(save_folder_path):
            print("이미 {}가 처리되었습니다".format(actor_folder))
            continue
        else:
            createFolder(save_folder_path)

        actor_folder_path = root_path + "/raw_image/{}".format(actor_folder)
        actor_img_list = os.listdir(actor_folder_path)
        collect_img = []
        for actor_img in actor_img_list:
            print(actor_img, "처리 시작")
            if checkImgExtension(actor_img):
                img_path = actor_folder_path + "/{}".format(actor_img)
                actor_face_img = cv2.imread(img_path)
                # 얼굴만 detection해서 잘라냄
                try:
                    detect_faces = RetinaFace.detect_faces(actor_face_img, model=detect_model)
                except:
                    print("다음 사진 이동")
                    continue
                if (type(detect_faces) == dict):
                    for key in detect_faces.keys():
                        face = detect_faces[key]
                        facial_area = face['facial_area']
                        crop_face = cropFullFace(actor_face_img, facial_area)
                        crop_face = cv2.resize(crop_face, (112, 112))
                        collect_img.append(crop_face)
        # 코사인 유사도로 중복 제거
        new_collect_img = filterCosineSimilarityForImage(collect_img, recognition_model)
        print("{}개 중 {}개 추출".format(len(collect_img), len(new_collect_img)))
        #저장
        for idx, img in enumerate(new_collect_img):
            plt.imsave(save_folder_path + "/{}.jpg".format(idx), img)
        print("{} 처리 완료: {}".format(actor_folder, round(time.time() - stime)))

def pathToEmbedding(img_path, recognition_model, img_return=False):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    process_img, process_flip_img = preprocessImage(img)
    embedding = imgToEmbedding(process_img, recognition_model, img_flip=process_flip_img)
    if img_return:
        return embedding, img
    return embedding


def filterActorFace(root_path, actor_name, recognition_model):
    seed_path = root_path + "/seed/{}.jpg".format(actor_name)
    faces_folder_path = root_path + "/face/{} 아역_img".format(actor_name)

    # 시드 이미지 임베딩
    seed_embedding = pathToEmbedding(seed_path, recognition_model)
    print("{} 시드 세팅 완료".format(actor_name))

    #나머지 임베딩 하면서 거리 비교
    img_distance_list = []
    face_img_list = os.listdir(faces_folder_path)
    for img_name in face_img_list:
        if checkImgExtension(img_name):
            stime = time.time()
            actor_img_path = faces_folder_path + "/{}".format(img_name)
            embedding, img = pathToEmbedding(actor_img_path, recognition_model, img_return=True)
            distance = calculateDistance(embedding, seed_embedding)
            if distance < 1:
                img_distance_list.append([img, distance])
            print("{} 처리 시간: {}".format(img_name, round(time.time() - stime), 4))
    # 거리 순으로 정렬
    img_distance_list.sort(key=lambda x:x[1])

    # 최대 60개 필터
    extract_num = min(60, len(img_distance_list))
    img_distance_list = img_distance_list[:extract_num]
    directory = "/content/drive/MyDrive/face_recognition_modules/crawling/result/{}".format(actor_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    #저장
    for idx, img in enumerate(img_distance_list):
        plt.imsave(directory + "/{}.jpg".format(idx), img[0])