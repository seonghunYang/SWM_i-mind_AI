import numpy as np
from keras.preprocessing import image
from retinaface.commons import postprocess
import cv2, torch
def cropFace(img, area):
    img_copy = img.copy()
    left = area[0]
    top = area[1]
    right = area[2]
    bottom = area[3]
    return img_copy[top: bottom, left: right]


def alignFace(img, landmarks):
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    nose = landmarks["nose"]
    mouth_right = landmarks["mouth_right"]
    mouth_left = landmarks["mouth_left"]
    img = postprocess.alignment_procedure(img, right_eye, left_eye, nose)

    return img[:, :, ::-1]

def resizeFace(img, target_size=(112, 112)):
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
    img = cv2.resize(img, dsize)

    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]
    img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')

    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    return img

def alignImage(frame, faces):
    result = []
    for key in faces.keys():
        face = faces[key]
        facial_area = face['facial_area']
        # face crop
        crop_face = cropFace(frame, facial_area)
        # face align
        landmarks = face['landmarks']
        align_face = alignFace(crop_face, landmarks)
        #resize
        resize_face = resizeFace(align_face)
        result.append(resize_face)

    return result

def preprocessImage(img, flip=True):
    copy_img = img.copy()
    device = torch.device('cuda')
    copy_img = np.array(copy_img)
    copy_img = np.transpose(copy_img, (2, 0, 1))
    img_normal = torch.from_numpy(copy_img).to(device).unsqueeze(0).float()
    # normalize
    img_normal.div_(255).sub_(0.5).div_(0.5)
    # flip image
    if flip:
        copy_img = np.flip(copy_img, axis=2).copy()
        img_flip = torch.from_numpy(copy_img).to(device).unsqueeze(0).float()
        img_flip.div_(255).sub_(0.5).div_(0.5)

        return img_normal, img_flip

    return img_normal

def cropFullFace(img, area):
    img_copy = img.copy()
    left = area[0]
    top = area[1]
    right = area[2]
    bottom = area[3]
    center = [(top + bottom) // 2, (left + right) // 2]
    y_half_len = (bottom - top) // 2
    full_left = max(0, center[1] - y_half_len) 
    full_right = min(center[1] + y_half_len, len(img_copy[1])) 
    return img_copy[top: bottom, full_left: full_right, ::-1]