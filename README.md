# CatchNet
<p>서비스 Catch는 대한민국 어린이집 99%에 이미 설치되어 있는 CCTV 인프라에 최신 트랜드의 AI 컴퓨터 비전 기술인 얼굴 식별, 객체 추적 및 행동 인식 기술을 활용하여 자동으로 아이의 성장 지표와 원내 생활 등 다양하고 균형있는 정보를 제공한다.</p>

<br>

## 환경 구성
<p>


```
conda create -n CatchNet python=3.7 -y
conda activate CatchNet
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11 -c pytorch -c conda-forge -y
# install pytorch with the same cuda version as in your environment
# cuda_version=$(nvcc --version | grep -oP '(?<=release )[\d\.]*?(?=,)')
# conda install pytorch torchvision cudatoolkit=$cuda_version -c pytorch

conda install av -c conda-forge -y
conda install cython -y

gt clone https://git.swmgit.org/swm-12/12_swm11/i-mind.git
cd i-mind/CatchNet
pip install -e . # Other dependicies will be installed here
pip install -r requirements.txt
pip install tensorflow
```
</p>

<br>

<p>

본 저장소는 다음과 같은 오픈소스 코드들을 참고하여 작성되었다.
- [YOLOv5 + DeepSORT](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
- [Towards-Realtime-MOT(JDE)](https://github.com/Zhongdao/Towards-Realtime-MOT)
- [Multi-Camera-Person-Tracking-and-Re-Identification](https://github.com/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification)
- [AlphAction](https://github.com/MVIG-SJTU/AlphAction)
</p>