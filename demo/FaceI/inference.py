import cv2
import numpy as np
import torch
from sklearn.preprocessing import normalize

@torch.no_grad()
def imgToEmbedding(img, model, img_flip=None):
    embedding1 = model(img).cpu().detach().numpy()
    if img_flip != None:
        embedding2 = model(img_flip).cpu().detach().numpy()
        embedding1 = embedding1 + embedding2
    embedding1 = normalize(embedding1)

    return embedding1[0]

def calculateDistance(embedding1, embedding2):
    diff = np.subtract(embedding1, embedding2)
    distance = np.sum(np.square(diff))

    return distance

def identifyFace(embedding, db, threshold=False):
    return ""
    # min_distance = threshold
    # label = ''

    # for idx, db_embedding in enumerate(db['embedding']):
    #     distance = calculateDistance(embedding, db_embedding)

    #     if min_distance == False or distance <= min_distance:
    #         min_distance = distance
    #         label = db['labels'][idx]

    # return label