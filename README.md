# i-mind-net
<p>

  <div align='center'>
    <img src="https://drive.google.com/uc?export=view&id=1iuYrMebgLdJ7uIEd8D4KTjuSgyIB9RDr" alt="">
  </div>
</p>
<p>

  <div align='center'>
    <img src="https://drive.google.com/uc?export=view&id=1lhTuuC3Vzx-BXi4RfQFjYhWsO-FlP7o8" alt="">
  </div>
</p>

  <div align='center'>소프트웨어 마에스트로 12기 i-mind팀 - 서비스 i-mind의 핵심 AI 파이프라인</div>
</p>

## 소개
i-mind-net은 서비스 i-mind에서 영상을 입력받으면 영상 내 객체 별 행동과 표정(감정) 정보를 메타데이터로 추출하는 기능을 수행합니다. 해당 데이터를 통해 영상 내 등장했던 객체를 구별하여 어느 시간대에 어떤 행동과 감정을 가졌는지 등을 알 수 있습니다.

## 데모
```
cd ${i-mind-net path}/demo
python demo.py --s3 --bucket-name <bucket_name> --s3-file-key <s3_file_key> --reid --show-id
```

<br>

## 개발 및 운영 환경
<p>

- AWS AMI: Deep Learning AMI (Ubuntu 18.04) Version 51.0 - ami-0c68557d08492cc1f
- RAM: 최소 16GB
- GPU: RTX 3070 Laptop, Tesla T4
- OS: Ubuntu 18.04
- Tools: CUDA, Python, Anaconda, PyTorch, Tensorflow 2.0+, NVIDIA-Docker
</p>

<br>

## 전체 프로세스
<p>
  <div align="center">
    <figure>
        <img src="https://drive.google.com/uc?export=view&id=1iYWmfFo3YUyHv9JvUb9_QBbi4-rsT-Iu" alt="i-mind-net 전체 프로세스">
        <div align="center"><figcation>i-mind-net 전체 프로세스</figcation></div>
    </figure>
  </div>
</p>

<p>

*i-mind-net*은 크게 두 트랙으로 병렬 처리됩니다. 하나는 객체의 행동을 인식하기 위한 트랙이고, 다른 하나는 객체의 표정과 감정을 인식하기 위한 트랙입니다.
</p>
<p>

i-mind-net 구조와 기술에 대해 더 자세히 알고싶으시면 [기술 문서](https://github.com/DrMaemi/i-mind-net/blob/master/DESCRIPTION.md)를 참조하세요.
</p>


<br>

## 환경 구성
<p>


```
conda create -n i-mind-net python=3.7 -y
conda activate i-mind-net
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch -c conda-forge -y

conda install av -c conda-forge -y
conda install cython -y

git clone https://git.swmgit.org/swm-12/12_swm11/i-mind-net.git
cd i-mind/i-mind-net
pip install -e .
pip install -r requirements.txt
./download_weights.sh
```
</p>

<br>

## 디렉토리 구조
```
$ tree -d -I "__pycache__.py|dataset|build"
.
|-- alphaction ### 행동 인식부 라이브러리
|   |-- config ### 딥러닝 네트워크 설정 파일
|   |-- csrc ### CPU, CUDA 환경 동작 설정 라이브러리
|   |   |-- cpu
|   |   `-- cuda
|   |-- engine
|   |-- layers
|   |-- modeling
|   |   |-- backbone ### SlowFast 네트워크 아키텍처 및 구조 설정
|   |   |-- detector ### SlowFast 네트워크 Action Detection 라이브러리
|   |   `-- roi_heads ### SlowFast 네트워크 마지막 단 softmax, postprocess, 손실 함수
|   |       `-- action_head
|   |-- solver
|   |-- structures
|   `-- utils
|-- config_files
|-- demo # i-mind-net 실행부
|   |-- FaceI # 얼굴 식별 모듈 라이브러리
|   |   |-- backbones
|   |   |-- configs
|   |   |-- eval
|   |   |-- retinaface
|   |   |   |-- commons
|   |   |   `-- model
|   |   `-- utils
|   `-- dummy
|-- detector ### 객체 탐지·추적 모델 구현부
|   |-- Yolov5_DeepSort_Pytorch
|   |   |-- MOT16_eval
|   |   |-- deep_sort_pytorch
|   |   |   |-- configs
|   |   |   |-- deep_sort
|   |   |   |   |-- deep
|   |   |   |   |   `-- checkpoint
|   |   |   |   `-- sort
|   |   |   `-- utils
|   |   `-- yolov5
|   |       |-- models
|   |       |   `-- hub
|   |       |-- utils
|   |       |   |-- aws
|   |       |   |-- flask_rest_api
|   |       |   |-- google_app_engine
|   |       |   `-- wandb_logging
|   |       `-- weights
|   |-- nms
|   |   `-- src
|   |-- tracker ### 행동 인식 모델과의 연동 인터페이스 구현부
|   |   |-- cfg
|   |   |-- tracker
|   |   `-- utils
|   `-- yolo
|       `-- cfg
|-- indicator ### 서비스 지표 산출 머신러닝 모델 MLOps 코드
|-- projector ### 임베딩 벡터 시각화
|-- tools
|   `-- ava ### AVA 데이터셋 구축·전처리
`-- torchreid ### 객체 재식별 라이브러리
    |-- data
    |   `-- datasets
    |       |-- image
    |       `-- video
    |-- engine
    |   |-- image
    |   `-- video
    |-- losses
    |-- metrics
    |   `-- rank_cylib
    |-- models
    |-- optim
    `-- utils
```

<br>

## Model Zoo
### Pre-trained Models
<p>i-mind-net는 SlowFast-R101을 backbone으로 사용하여 전이학습시킨 모델을 사용합니다. 커스텀 데이터셋을 학습시켜 본인만의 모델을 만들고싶은 분은 아래 backbone 모델을 활용하세요.</p>
<p>

| backbone | pre-train | frame length | sample rate | top-1 | top-5 | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| SlowFast-R50 | Kinetics-700 | 4 | 16 | 66.34 | 86.66 | [[link]](https://drive.google.com/file/d/1bNcF295jxY4Zbqf0mdtsw9QifpXnvOyh/view?usp=sharing) |
| SlowFast-R101 | Kinetics-700 | 8 | 8 | 69.32 | 88.84 | [[link]](https://drive.google.com/file/d/1v1FdPUXBNRj-oKfctScT4L4qk8L1k3Gg/view?usp=sharing) |
</p>

### Model 경로
<p>


분류 | 모델(다운로드 링크) | 경로
--- | --- | ---
객체 탐지 | [yolov3-spp.weights](https://drive.google.com/open?id=1T13mXnPLu8JRelwh60BRR21f2TlGWBAM) | ./data/models/detector_models/
객체 추적 | [crowdhuman_yolov5m.pt](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing) | ./detector/Yolov5_DeepSort_Pytorch/yolov5/weights/
객체 재추적(Re-ID) | [model.pth](https://drive.google.com/file/d/1_LoiFYlsVu3ervIidYIMopodyBLswmi4/view?usp=sharing) | ./demo/model_data/models/
행동 인식 | [resnet101_8x8f_denseserial](https://drive.google.com/file/d/1DKHo0XoBjrTO2fHTToxbV0mAPzgmNH3x/view?usp=sharing) | ./data/models/aia_models/
</p>

<br><br>


## 참조
<p>

- [YOLOv5 + DeepSORT](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
- [Towards-Realtime-MOT(JDE)](https://github.com/Zhongdao/Towards-Realtime-MOT)
- [Multi-Camera-Person-Tracking-and-Re-Identification](https://github.com/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification)
- [AlphAction](https://github.com/MVIG-SJTU/AlphAction)
- [AWS Rekognition](https://aws.amazon.com/ko/rekognition/)
</p>