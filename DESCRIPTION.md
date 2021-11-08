# i-mind-net 기술 문서

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

## 1. 행동 인식
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

<p>

한 편 객체별로 정보를 모으기 위해서는 객체 추적이 실패하는 경우를 최대한 만회하는 것이 중요한데, 객체 재추적부는 영상 내 등장 객체가 영상 밖으로 나갔다 들어오거나(이하 재등장), 오클루전 등에 의한 추적 실패를 방지하여 객체 별 분석 검출이 지속되도록 재추적 기술을 사용합니다.
</p>

<p>
  <div align="center">
    <figure>
        <img src="https://drive.google.com/uc?export=view&id=1r6S5Y5Gi4WGLgXs6qFPPCp6jhSj84ZoT" alt="객체 재추적 프로세스 자세히">
        <div align="center"><figcation>객체 재추적 아키텍처</figcation></div>
    </figure>
  </div>
</p>

<p>

먼저 등장했던 객체 ID 정보와 single linkage 계층적 군집화 기법을 적용하여 추적에 실패했던 객체를 재추적합니다. 특징점 벡터를 추출할 때 객체 경계 박스 이미지 한 개씩 추출하기보다 큐에 쌓아두어 분석 시스템의 CPU-GPU 메모리 크기에 맞게 배치 처리를 수행하면 메모리 복사 및 CPU-GPU 간 I/O 빈도가 낮아져 처리 속도가 월등히 빨라지도록 개선했습니다. 또한 객체 경계 박스 이미지에서 특징점 벡터를 추출한 뒤, 더 이상 사용하지 않는 이미지들은 보관하지 않고 버림으로써 메모리 요구사항을 대폭 감소시켰습니다.
</p>

<p>
  <div align="center">
    <figure>
        <img src="https://drive.google.com/uc?export=view&id=1lIlMbo4CxaKqnKueFhvkR9ichO78e_9f" alt="4번 객체가 영상에서 사라졌다가 재등장해도 추적 가능">
        <div align="center"><figcation>4번 객체가 영상에서 사라졌다가 재등장해도 추적 가능</figcation></div>
    </figure>
  </div>
</p>

<p>
  <div align="center">
    <figure>
        <img src="https://drive.google.com/uc?export=view&id=10aFdRQTg62AEAOdWUbcf9JX7Ow9waD3R" alt="5번 객체가 가려졌어도(오클루전) 추적 가능">
        <div align="center"><figcation>5번 객체가 가려졌어도(오클루전) 추적 가능</figcation></div>
    </figure>
  </div>
</p>

<p>
  <div align="center">
    <figure>
        <img src="https://drive.google.com/uc?export=view&id=13WnTnwsQMhZZm4LjjkFYYo0uoTLjg7DO" alt="Re-ID 임베딩 벡터 프로젝터">
        <div align="center"><figcation>임베딩 공간의 벡터들이 군집(1, 2, 3, 4, 5, 60, 64번 객체에 따라)을 이룸</figcation></div>
    </figure>
  </div>
</p>


## 2. 표정 인식
<p>

프레임 내 인식한 얼굴에서 특징점들을 추출하고 표정을 인식한다. 인식한 표정을 객체별 ID에 연동하여 시간에 따라 특정 객체가 어떤 표정(감정)을 지녔었는지 알 수 있도록 합니다.
</p>
<p>

탐지 가능한 표정 종류는 총 8가지: 1. 놀람, 2. 혼란, 3. 평온, 4. 기쁨, 5. 화남, 6. 질색·역겨움, 7. 슬픔, 8. 무서움·공포 입니다.
</p>

<p>
  <div align="center">
    <figure>
        <img src="https://drive.google.com/uc?export=view&id=1so9umsNi9tGjWJIDwEBIymOC6af9JkEU" alt="표정 인식부 기능 수행 순서도">
        <div align="center"><figcation>표정 인식부 기능 수행 순서도</figcation></div>
    </figure>
  </div>
</p>

<p>

i-mind-net에서는 행동 인식 결과와 연동할 때 IoF(Intersection over Face) 알고리즘을 사용합니다.
</p>

<p>
  <div align="center">
    <figure>
        <img src="https://drive.google.com/uc?export=view&id=1XOCZChfpg5ZagMDXbfb11UH4iRHiyPGT" alt="IoF 기본 알고리즘 수식">
        <div align="center"><figcation>IoF 기본 알고리즘 수식</figcation></div>
    </figure>
  </div>
</p>

<p>

인식한 표정의 얼굴 위치와, 행동 인식 프로세스 종료 후 알고 있는 객체 위치 각각의 경계 박스 면적을 바탕으로 위와 같은 수식을 통해 IoF 면적을 구하여 가장 면적이 큰 객체 ID를 표정에 부여합니다.
</p>
<p>

위 수식만으론 객체가 겹쳐있을 경우 원하는 대로 동작하지 않으므로, IoF 면적이 가장 크게 나온 객체가 여럿일 경우 영상 내 원근에 따라 적절한 객체 ID를 부여하도록 [그림 27]과 같이 동적 계획법과 탐욕법 알고리즘을 2차원 행렬 자료구조를 통해 구현하여 약점을 보완했습니다.
</p>

<p>
  <div align="center">
    <figure>
        <img src="https://drive.google.com/uc?export=view&id=10g2174rkfgKtJuAP1vdzyQWcBmLfRWQz" alt="IoF 보완 알고리즘 의사 코드">
        <div align="center"><figcation>IoF 보완 알고리즘 의사 코드</figcation></div>
    </figure>
  </div>
</p>

<p>
  <div align="center">
    <figure>
        <img src="https://drive.google.com/uc?export=view&id=1JvWHXG1Kq7DaRwyN_q-9JPIIMq9H2F3r" alt="IoF 보완 알고리즘이 극복할 수 있는 상황">
        <div align="center"><figcation>IoF 기본 알고리즘만으로 해결되는 상황(왼쪽)과 보완 알고리즘이 필요한 상황(오른쪽)</figcation></div>
    </figure>
  </div>
</p>