# i-mind-net
<p>

소프트웨어 마에스트로 12기 i-mind팀 서비스 *i-mind*의 핵심 AI 파이프라인
</p>

<br>

## 동작 환경
<p>

- RAM: 최소 16GB
- GPU: GTX 960 이상
- OS: Ubuntu 18.04
- Tools: CUDA, Python, Anaconda, PyTorch, Tensorflow 2.0+
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

### 1. 행동 인식
<p>

행동 인식을 위해서 Object Detector & Tracker, Video Model(SlowFast 네트워크)과 Action Detector를 거칩니다. 비디오를 여러 개의 클립으로 자르고, 클립의 각 프레임을 통해 Object Detector & Tracker에서 객체의 위치 정보와 추적 정보를 얻습니다. 동시에 클립이 Video Model에 입력되어 클립 전체의 특징 벡터들을 추출합니다.
</p>

<p>
  <div align="center">
    <figure>
        <img src="https://drive.google.com/uc?export=view&id=1yd1MTwPDI2VulU0jUwwl66tJ17IY5tBG" alt="행동 인식 프로세스">
        <div align="center"><figcation>행동 인식 프로세스</figcation></div>
    </figure>
  </div>
</p>

<p>

추출한 특징 벡터들로부터 객체의 위치 정보를 바탕으로 객체들의 특징 벡터들을 RoI Align 후 사람 객체의 특징 벡터를 통해 비디오의 temporal 정보를 업데이트합니다. 최종적으로 업데이트된 메모리 특징 벡터와 사람 객체의 특징 벡터, 사물의 특징 벡터를 상호작용 통합(Interaction Aggregation, IA) 네트워크에 입력하여 행동을 인식합니다.
</p>

<br>

## 환경 구성
<p>


```
conda create -n i-mind-net python=3.7 -y
conda activate i-mind-net
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11 -c pytorch -c conda-forge -y
# install pytorch with the same cuda version as in your environment
# cuda_version=$(nvcc --version | grep -oP '(?<=release )[\d\.]*?(?=,)')
# conda install pytorch torchvision cudatoolkit=$cuda_version -c pytorch

conda install av -c conda-forge -y
conda install cython -y

git clone https://git.swmgit.org/swm-12/12_swm11/i-mind-net.git
cd i-mind/i-mind-net
pip install -e . # Other dependicies will be installed here
pip install -r requirements.txt
```
</p>

<br>

## 디렉토리 구조
<p>

  <div align="center">
    <figure>
        <img src="https://drive.google.com/uc?export=view&id=1vAnimz8Bojgd-e-xl-_1IbHot-owkzA1" alt="CatchNet 디렉토리 구조">
        <div align="center"><figcation>CatchNet 디렉토리 구조</figcation></div>
    </figure>
  </div>
</p>

<br>

## Model Zoo
### Pre-trained Models
<p>커스텀 데이터셋을 학습시켜 본인만의 모델을 만들고싶은 분은 아래 Backbone 모델을 활용하세요.</p>
<p>

| backbone | pre-train | frame length | sample rate | top-1 | top-5 | model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| SlowFast-R50 | Kinetics-700 | 4 | 16 | 66.34 | 86.66 | [[link]](https://drive.google.com/file/d/1bNcF295jxY4Zbqf0mdtsw9QifpXnvOyh/view?usp=sharing) |
| SlowFast-R101 | Kinetics-700 | 8 | 8 | 69.32 | 88.84 | [[link]](https://drive.google.com/file/d/1v1FdPUXBNRj-oKfctScT4L4qk8L1k3Gg/view?usp=sharing) |
</p>

### Models
<p>바로 데모를 수행하고 싶으신 분은 아래 모델을 다운받아 안내드리는 경로에 위치시켜주세요.</p>
<p>


분류 | 모델(다운로드 링크) | 경로
:-: | :-: | :-:
객체 탐지 | [yolov3-spp.weights](https://drive.google.com/open?id=1T13mXnPLu8JRelwh60BRR21f2TlGWBAM) | ./data/models/detector_models/
객체 추적 | [crowdhuman_yolov5m.pt](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing) | ./detector/Yolov5_DeepSort_Pytorch/yolov5/weights/
객체 재추적(Re-ID) | [model.pth](https://drive.google.com/file/d/1_LoiFYlsVu3ervIidYIMopodyBLswmi4/view?usp=sharing) | ./demo/model_data/models/
행동 인식 | [resnet101_8x8f_denseserial](https://drive.google.com/file/d/1DKHo0XoBjrTO2fHTToxbV0mAPzgmNH3x/view?usp=sharing) | ./data/models/aia_models/
</p>

<br><br>

<p>

본 저장소는 다음과 같은 오픈소스 코드들을 참고하여 작성되었습니다.
- [YOLOv5 + DeepSORT](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
- [Towards-Realtime-MOT(JDE)](https://github.com/Zhongdao/Towards-Realtime-MOT)
- [Multi-Camera-Person-Tracking-and-Re-Identification](https://github.com/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification)
- [AlphAction](https://github.com/MVIG-SJTU/AlphAction)
</p>