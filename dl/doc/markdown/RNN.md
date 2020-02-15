## 논문 검색

- https://paperswithcode.com

# RNN 정리

- 순환 신경망 (Recurrent Neural Network)
- RNN은 은닉층의 노드에서 활성화 함수를 통해 나온 결과값을 출력층 방향으로 보내면서, 다시 은닉층 노드의 다음 계산의 입력으로 보내는 특징을 가진다(두군데로 보낸다)
- 피드 포워드 신경망(Feed Forward Neural Network)
  - 일반적으로 신경망에서 은닉층에서 활성화 함수를 지난 출력값은 출력층으로만 간다 => 순방향

<img src="C:\Users\admin\Documents\GitHub\pengsoo\dl\data\rnn_f_0.png" style="zoom:150%;" />

- 그림 설명
  - x는 입력층의 입력백터
  - y는 출력층의 출력백터
  - 실제 수식에서는 편향값 b도 존재하나 그림에서 생략
  - cell: 은닉층에서 활성화 함수를 통해 결과를 내보내는 역할. 이전값의 값을 기억하려는(t-1를 입력으로 받는다)  메모리 역할도 하기 때문에 메모리 셀, RNN Cell이라고 표현
  - 메모리 셀은 각각의 시점(time step)에서 이전 시점에서 들어온 값(이전 시점의 은닉층의 메모리셀에서 나온 값)을 자신의 입력값으로 사용하는 재귀적인 활동을 수행한다
  - 현재 시점 t
  - 현재 시점 t에서 현재 시점의 메모리셀이 가진 값은 과거의 메모리 셀의 값들의 영향을 받는다
  - 메모리셀이 출력층 방향으로 혹은 다음 시점(t+1)로 내보내는 값 = 은닉 상태(hidden state)

![](C:\Users\admin\Documents\GitHub\pengsoo\dl\data\rnn_f_1.png)

- 그림 설명
  - 좌측
    - 사이클을 표현하는 화살표를 사용하여서 표현
    - 재귀형태
  - 우측
    - 여러 시점(time step)을 기준으로 펼쳐서 표현
    - 시간의 흐름에 따라 표현

- 요약
  - 입력층 -> 입력 백터
  - 출력층 -> 출력 백터 
  - 은닉층 -> 은닉 상태
  - 위 그림의 모든 사각형 박스는 백터를 표현

![](C:\Users\admin\Documents\GitHub\pengsoo\dl\data\rnn_f_2.png)

- 그림 
  - 입력백터 차원 4 / 은닉상태 크기 2 / 출력백터 차원 2


- RNN의 망구성 방식
  - 하나의 입력에서 여러개 출력(one-to-many)
    - 이미지 캡셔닝, 하나의 이미지 입력에 대헤서 제목(단어들의 나열 => 시퀀스), 내용을 출력하는 작업
  - 여러개 입력에서 하나의 출력(many-to-one)
    - 입력 문서가 긍정적인지 부정적인지 판별 : 감성분류(sentiment classification)
    - 스팸 메일 분류
  - 다 대 다 (many-to-many)
    - 입력문장에 대해 문장을 만들어준다 => 챗봇
    - 입력문장으로부터 번역된 문장을 출력=> 기계번역
    - 입력대비 동시에 출력되는 방식, 입력후 망 후반에서 출력이 되는 방식

![](C:\Users\admin\Documents\GitHub\pengsoo\dl\data\rnn_f_3.png)

- 그림 설명
  - 현재 시점 => t
  - 현재 시점의 은닉상태 => ht
  - ht를 계산하기 위해서 2개의 입력이 필요하고, 각각의 입력에는 2개의 가중치(W)가 필요하다
    - 하나는 현재 시점의 입력층을 위한 가중치 Wx
    - 또 하나는, 이전 시점의 은닉 상태값 ht-1을 위해 사용한 가중치 Wh
  - 식
    - 메모리셀, 은닉상태 : ht=tanh( WxXt + WhHt-1 + b(2종류를 통틀어서표현) )
    - 출력층 : Yt = 활성화함수( Wyht + b)

![](C:\Users\admin\Documents\GitHub\pengsoo\dl\data\rnn_f_5.png)

- BPTT(Backpropagation through time)
  - RNN은 전체 시점에 대해서 네트워크를 펼친다음, 역전파를 이용하여 모든 시점의 가중치 공유
  - 절차
    - 순전파를 통해서 모든 시점에 대한 시퀀스를 출력
    - 각 시점의 손실(loss)에 보고, 가중치(w)를 보정한다
- 깊은 순환 신경망(Deep RNN)
  - 순환 신경망에서 은닉층이 1개더 추가 되었다. 이를 통해 deep를 표현. 
  - cell을 하나 더 추가
  - 학습 능력 향상

![](C:\Users\admin\Documents\GitHub\pengsoo\dl\data\rnn_f_6.png)

- 양방향 순환 신경망(Bidirectional RNN)
  - 시점 t에서 출력값을 예측할때, 이전 시점의 데이터뿐만 아니라, 이후 데이터(미래) 데이터로 출력값을 예측할수 있다라는 가정
  - 실제로도, 과거 시점의 데이터를 참고하여 정답을 주로 예측하지만, 특정 문제에 대해서 미래의 값이 어떤 단서가 되기도 한다.
  - 결론
    - 이전 시점의 데이터 + 이후 시점의 데이터를 활용하여 종합적으로 예측(출력)
    

![](C:\Users\admin\Documents\GitHub\pengsoo\dl\data\rnn_f_7.png)

- 그림 설명
  - 2개의 메모리 셀을 사용
  - 첫번째 셀
    - 주황색
    - 앞 시점(t-1)의 은닉상태(hidden state)를 전달받아서 은닉 상태를 계산
  - 두번째 셀
    - 초록색
    - 뒤 시점(t+1)의 은닉상태(hidden state)를 전달받아서 은닉 상태를 계산
  - 두 개 값을 이용하여 출력층의 출력값을 예측시 사용한다

- 양방향 깊은 순환 신경망(Bidirectional Deep RNN)
  - 양방향 순환 신경망에 deep을 더한것



![](C:\Users\admin\Documents\GitHub\pengsoo\dl\data\rnn_f_8.png)

```python
!pip install konlpy

!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!pip3 install JPype1-py3

! bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

import os
os.chdir('/tmp/')
!curl -LO https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.1.tar.gz
!tar zxfv mecab-0.996-ko-0.9.1.tar.gz
os.chdir('/tmp/mecab-0.996-ko-0.9.1')
!./configure
!make
!make check
!make install

import os
os.chdir('/tmp')
!curl -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.0.1-20150920.tar.gz
!tar -zxvf mecab-ko-dic-2.0.1-20150920.tar.gz
os.chdir('/tmp/mecab-ko-dic-2.0.1-20150920')
!./autogen.sh
!./configure
!make
# !sh -c 'echo "dicdir=/usr/local/lib/mecab/dic/mecab-ko-dic" > /usr/local/etc/mecabrc'
!make install


# install mecab-python
import os
os.chdir('/content')

!git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git
os.chdir('/content/mecab-python-0.996')

!python3 setup.py build
!python3 setup.py install
```



# 영화 댓글을 이용한 감성 분류
- 최종 산출물 : 댓글을 작성하여 입력하면 긍정/부정을 판단

- 학습할때 노트를 GPU 변경하고 수행

  

## 1. 패키지로드

```python
# 엔진, 행렬연산용
import torch

# 층생성, 활성화, 최적화, 손실함수 
import torch.nn as nn
import torch.optim as optim

# 텍스트 로드, 전처리용
from torchtext.data import Field, TabularDataset, Iterator

# 형태소 분석기(한글때문에)
from konlpy.tag import Mecab
```

## 2. 데이터 로드 및 전처리
- torchtext의 데이터 전처리 절차를 수행하는 클레스 
  - 데이터 정의 
  - 데이터셋 구성
  - 사전(단어장) 생성
  - 데이터로더 생성

```python

```

## 3. 신경망 구축

```python

```

## 4. 훈련 및 테스트 전용 모듈 구성

- torchtext의 데이터 전처리 절차를 수행하는 클레스 
  - 데이터 정의 

```python

```

## 5. 학습

- torchtext의 데이터 전처리 절차를 수행하는 클레스 
  - 데이터 정의 

```python

```

## 6. 실전 테스트(실제 댓글 작성 후 예측)

- torchtext의 데이터 전처리 절차를 수행하는 클레스 
  - 데이터 정의 

```python

```

