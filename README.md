# i-mind-net

<p>i-mind-net은 서비스 Catch에서 사용하는 AI 파이프라인 네트워크이다. 최신 트랜드의 AI 컴퓨터 비전 기술인 얼굴 식별, 객체 추적 및 행동 인식 기술을 활용하여 어린이집 CCTV 환경에서 자동으로 아이의 성장 지표와 원내 생활 등 다양하고 균형있는 정보를 제공한다.</p>

<br>

## 동작 환경
<p>

- RAM: 최소 16GB
- GPU: GTX 960 이상
- OS: Ubuntu 18.04
- Tools: CUDA, Python, Anaconda, PyTorch, Tensorflow 2.0+
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