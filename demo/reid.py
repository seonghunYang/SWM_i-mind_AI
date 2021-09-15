from torchreid.data.transforms import build_transforms
import cv2
from PIL import Image
import torchreid
import torch
import os
from torchreid import metrics
from timeit import time
from annoy import AnnoyIndex


class REID:
    def __init__(self):
        self.model = torchreid.models.build_model(
                name='resnet50',
                num_classes=1,#human
                loss='softmax',
                pretrained=True,
                use_gpu = True
            )
        torchreid.utils.load_pretrained_weights(self.model, 'model_data/models/model.pth')
        self.model = self.model.cuda()
        self.optimizer = torchreid.optim.build_optimizer(
                self.model,
                optim='adam',
                lr=0.0003
            )
        self.scheduler = torchreid.optim.build_lr_scheduler(
                self.optimizer,
                lr_scheduler='single_step',
                stepsize=20
            )
        _, self.transform_te = build_transforms(
            height=256, width=128,
            random_erase=False,
            color_jitter=False,
            color_aug=False
        )
        self.dist_metric = 'euclidean'
        self.model.eval()

    def _extract_features(self, input):
        self.model.eval()
        return self.model(input)
    
    def _features(self, imgs):
        f = []
        total_num = len(imgs) # 특징점을 추출하고자 하는 이미지 총 개수
        batch_size = 10 # batch size. 메모리 용량에 맞게 조정 가능
        loop_num = total_num//batch_size

        for l in range(loop_num):
            start = l*batch_size
            end = start+batch_size
            batch_imgs = []
            features = []
            for i in range(start, end):
                imgs[i] = Image.fromarray(imgs[i].astype('uint8')).convert('RGB')
                imgs[i] = self.transform_te(imgs[i])
                imgs[i] = torch.unsqueeze(imgs[i], 0)
                batch_imgs.append(imgs[i].cuda())
            ft_imgs = torch.cat(batch_imgs, 0)
            # print(f"\nft_imgs.size() -> {ft_imgs.size()}") # (batch_size x 3 x width x height), width, height는 uint8?
            features = self._extract_features(ft_imgs)
            del ft_imgs
            torch.cuda.empty_cache()
            features = features.data.cpu()
            f.append(features)
        
        # Process last loop
        start = loop_num*batch_size
        batch_imgs = []
        features = []
        for i in range(start, total_num):
            imgs[i] = Image.fromarray(imgs[i].astype('uint8')).convert('RGB')
            imgs[i] = self.transform_te(imgs[i])
            imgs[i] = torch.unsqueeze(imgs[i], 0)
            batch_imgs.append(imgs[i].cuda())
        ft_imgs = torch.cat(batch_imgs, 0)
        # print(f"\nft_imgs.size() -> {ft_imgs.size()}") # (batch_size_remains x 3 x width x height), width, height는 uint8?
        features = self._extract_features(ft_imgs)
        del ft_imgs
        torch.cuda.empty_cache()
        features = features.data.cpu()
        f.append(features)

        f = torch.cat(f, 0)
        return f # (n x 2048)

    def compute_distance(self, qf, gf):
        distmat = metrics.compute_distance_matrix(qf, gf, self.dist_metric)
        # print(distmat.shape)
        return distmat.numpy()

if __name__ == '__main__':
    reid = REID()



