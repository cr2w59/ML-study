# 오차 역전파

- Back propagation
- 경사 하강법의 확장 개념

### 오차 수정 비교

- 단일 퍼셉트론에서는 경사 하강법으로 오차가 최소인 점을 찾아 최초 임의의 가중치를 점차 조정해나감
- 다층 퍼셉트론에서는 은닉층으로 인해 출력층에서 거슬러 올라가며 최적화를 진행함

## 구동 방식

1. 임의의 초기 가중치(`W`)를 준 뒤 결과(`y`)를 계산

2. 계산 결과와 내가 원하는 값 사이의 오차를 구함

3. 경사 하강법을 이용해 바로 앞 가중치를 오차가 작아지는 방향으로(기울기가 0이 되는 방향) 업데이트

4. 최소의 오차를 구할 때까지 위 과정을 반복

   - = 가중치에서 기울기를 빼도 값의 변화가 없을 때까지 
     $$
     W(t+1)=Wt-\frac{\partial오차}{\partial W}
     $$

   - 새 가중치`W(t+1)`는 현 가중치`W(t)`에서 가중치에 대한 기울기를 뺀 값

## 오차 역전파를 이용한 신경망 구현

1. 환경 변수 지정: input과 label의 데이터셋, 학습률, 활성화 함수, 가중치 선언
2. 신경망 실행: 초기값을 입력해 활성화 함수와 가중치를 거친 결과값이 나옴
3. 결과를 실제 값과 비교: 오차 측정
4. 은닉층과 출력층의 가중치 수정(반복)
5. 결과 출력

- xor_backpropagation.py

```python
import random, numpy as np

random.seed(777)
ITERATIONS = 5000 #실행횟수
LR = 0.1  #학습률
MO = 0.4  #모멘텀 계수

## 활성화 함수
# 1. 시그모이드
def sigmoid(x, derivative=False):
  if(derivative==True):
    return x * (1 - x)  #미분o
  return 1 / (1 + np.exp(-x)) #미분x

# 2. tanh
def tanh(x, derivative=False):
  if(derivative==True): #미분o
    return 1 - x ** 2
  return np.tanh(x) #미분x

# 가중치 배열을 만드는 함수
def makeMatrix(I, J, fill=0.0):
  mat = list()
  for i in range(I):
    mat.append([fill] * J)  
  return mat

## 신경망 실행
class NeuralNetwork:
  # 초기값 지정
  def __init__(self, num_i, num_h, num_o, bias=1):
    # 입력값(num_i), 은닉층의 초기값(num_h), 출력층의 초기값(num_o), 바이어스
    self.num_i = num_i + bias
    self.num_h = num_h
    self.num_o = num_o

    # 활성화 함수 초기값
    self.activation_input = [1.0] * self.num_i
    self.activation_hidden = [1.0] * self.num_h
    self.activation_output = [1.0] * self.num_o

    # 가중치 입력 초기값
    self.weight_in = makeMatrix(self.num_i, self.num_h)
    for i in range(self.num_i):
      for j in range(self.num_h):
        self.weight_in[i][j] = random.random()

    # 가중치 출력 초기값
    self.weight_out = makeMatrix(self.num_h, self.num_o)
    for j in range(self.num_h):
      for k in range(self.num_o):
        self.weight_out[j][k] = random.random()
    
    # 모멘텀 SGD를 위한 이전 가중치 초기값
    self.gradient_in = makeMatrix(self.num_i, self.num_h)
    self.gradient_out = makeMatrix(self.num_h, self.num_o)


  # 업데이트 함수
  def update(self, inputs):
    if len(inputs) != self.num_i - 1:
      raise ValueError('잘못된 입력')

    # 입력층의 활성화 함수
    for i in range(self.num_i - 1):
      self.activation_input[i] = inputs[i]

    # 은닉층의 활성화 함수
    for j in range(self.num_h):
      sum = 0.0
      for i in range(self.num_i):
        sum = sum + self.activation_input[i] * self.weight_in[i][j]
      # 활성화 함수 선택(미분x)
      self.activation_hidden[j] = tanh(sum, False)
      # self.activation_hidden[j] = sigmoid(sum, False)

    # 출력층의 활성화 함수
    for k in range(self.num_o):
      sum = 0.0
      for j in range(self.num_h):
        sum = sum + self.activation_hidden[j] * self.weight_out[j][k]
      # 활성화 함수 선택(미분x)
      self.activation_output[k] = tanh(sum, False)
      # self.activation_output[k] = sigmoid(sum, False)

    return self.activation_output[:]

  # 역전파 실행
  def backPropagate(self, targets):
    # 델타 출력 게산
    output_deltas = [0.0] * self.num_o
    for k in range(self.num_o):
      error = targets[k] - self.activation_output[k]
      # 활성화 함수 선택(미분o)
      output_deltas[k] = tanh(self.activation_output[k], True) * error
      # output_deltas[k] = sigmoid(self.activation_output[k], True) * error
    
    # 은닉 노드의 오차 함수
    hidden_deltas = [0.0] * self.num_h
    for j in range(self.num_h):
      error = 0.0
      for k in range(self.num_o):
        error = error + output_deltas[k] * self.weight_out[j][k]
      # 활성화 함수 선택(미분o)
      hidden_deltas[j] = tanh(self.activation_hidden[j], True) * error
      # hidden_deltas[j] = sigmoid(self.activation_hidden[j], True) * error

    # 출력 가중치 업데이트
    for j in range(self.num_h):
      for k in range(self.num_o):
        gradient = output_deltas[k] * self.activation_hidden[j]
        v = MO * self.gradient_out[j][k] - LR * gradient
        self.weight_out[j][k] += v
        self.gradient_out[j][k] = gradient

    # 입력 가중치 업데이트
    for i in range(self.num_i):
      for j in range(self.num_h):
        gradient = hidden_deltas[j] * self.activation_input[i]
        v = MO * self.gradient_in[i][j] - LR * gradient
        self.weight_in[i][j] += v
        self.gradient_in[i][j] = gradient

    # 오차 계산(최소 제곱법)
    error = 0.0
    for k in range(len(targets)):
      error = error + 0.5 * (targets[k] - self.activation_output[k]) ** 2

    return error

  # 학습
  def train(self, patterns):
    for i in range(ITERATIONS):
      error = 0.0
      for p in patterns:
        inputs = p[0]
        targets = p[1]
        self.update(inputs)
        error = error + self.backPropagate(targets)
      if i % 500 == 0:
        print('error: %-.5f' % error)
  
  # # 테스트
  # def test(self, patterns):
  #   for p in patterns:
  #     print(p[0], '->', self.update(p[0]))

  # 결과 출력
  def result(self, patterns):
    for p in patterns:
      print(f'Input: {p[0]}, Predict: {self.update(p[0])}')

def demo(num_i, num_h, num_o, gate='XOR'):
  # 입력값과 타깃값
  if gate == 'XOR':  
    DATASET = [
      [[0, 0], [0]], 
      [[0, 1], [1]], 
      [[1, 0], [1]], 
      [[1, 1], [0]], 
    ]
  elif gate == 'AND':
    DATASET = [
      [[0, 0], [0]], 
      [[0, 1], [0]], 
      [[1, 0], [0]], 
      [[1, 1], [1]], 
    ]
  elif gate == 'OR':
    DATASET = [
      [[0, 0], [1]], 
      [[0, 1], [1]], 
      [[1, 0], [1]], 
      [[1, 1], [0]], 
    ]
  # create a network with two input, two hidden, and one output nodes
  n = NeuralNetwork(num_i, num_h, num_o)
  # train it with some patterns
  n.train(DATASET)
  n.result(DATASET)

if __name__=='__main__':
  demo(2, 2, 1)
```



# 고급 경사 하강법

|    고급 경사 하강법     |                             개요                             |     개선사항      | Keras 사용법(Keras.optimizers.)                              |
| :---------------------: | :----------------------------------------------------------: | :---------------: | ------------------------------------------------------------ |
| 확률적 경사 하강법(SGD) | 랜덤하게 추출한 일부 데이터를 사용해 더 빨리, 자주 업데이트  |       속도        | SGD(lr=0.1)                                                  |
|    모멘텀(Momentum)     |            관성의 방향을 고려해 진동과 폭을 줄임             |      정확도       | SGD(lr=0.1, *momentum=0.9*)                                  |
| 네스테로프 모멘텀(MAG)  | 모멘텀이 이동시킬 방향으로 미리 이동해 그레이디언트를 계산. 불필요한 이동을 줄이는 효과 |      정확도       | SGD(lr=0.1, momentum=0.9, *nesterov=True*)                   |
|   아다그라드(Adagrad)   |  변수의 업데이트가 잦으면 학습률을 적게해 이동 보폭을 조절   |     보폭 크기     | *Adagrad*(*lr=0.01*, epsilon=1e-6)                           |
|  알엠에스프롭(RMSProp)  |               아다그라드의 보폭 민감도를 보완                |    보폭 민감도    | *RMSprop*(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)       |
|       아담(Adam)        |              모멘텀과 알엠에스프롭 방법을 합침               | 정확도, 보폭 크기 | *Adam*(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) |

## 1. 확률적 경사 하강법

- Stochastic Gradient Descent, SGD

$$
W(t+1)=W(t)-\eta\frac{\partial오차}{\partial W}\;\;단,x^{(i:i+n)}
$$

### 개념

- 경사 하강법은 불필요하게 많은 계산량으로 속도가 늦으며, 최적해를 찾기 전에 최적화 과정이 멈출 수도 있음 => 확률적 경사 하강법은 이런 속도의 단점을 보완함
- 전체가 아닌 랜덤하게 추출한 일부 데이터를 사용
- 빠르고 자주 업데이트 가능
- 중간 결과의 진폭이 크고 불안정해 보일 수도 있음
- 빠른 속도로 최적해에 근사한 값을 찾아내는 장점

### 구현

1. python

```python
self.weight[i] += learning_rate * gradient
```

2. Keras

```python
keras.optimizers.SGD(lr=0.1)
```



## 2. 모멘텀 SGD

- Momentum SGD

$$
V(t)=\gamma V(t-1)-\eta\frac{\partial 오차}{\partial W}\;\;단, \gamma는  \,모멘텀계수\\W(t+1)=W(t)+V(t)
$$

### 개념

- Momentum: 관성, 탄력, 가속도라는 뜻
- 경사 하강법에 탄력을 더해 주는 것
- 경사 하강법과 같이 매번 기울기를 구함
- 직전 수정 값과 방향을 참고해 같은 방향(+,-)으로 일정한 비율만 수정
  - 따라서 수정 방향이 + 한 번, - 한 번 지그재그로 일어나는 현상 줄어듦
- 이전 이동 값을 고려해 일정 비율만큼만 다음 값을 결정하므로 관성의 효과를 낼 수 있음

### 구현

1. python

```python
# m: 모멘텀 계수(앞서 구한 오차를 어느 정도 반영할지 정함)
v = m * v - learning_rate * gradient
self.weight[i] += v 
```

2. Keras
   - 확률적 경사 하강법에 모멘텀 계수만 추가해주면 됨

```python
keras.optimizers.SGD(lr=0.1, momentum=0.9)
```



## 3. 네스테로프 모멘텀

- Nesterov momentum

$$
V(t)=\gamma V(t-1)-\eta\frac{\partial 오차}{\partial(W+\gamma V(t-1))}\\W(t+1)=W(t)+V(t)
$$

### 개념

- gradient를 구할 때 먼저 `ΥV(t-1)`값을 더한 다음 계산
  - 이 단계로 인해 `V(t)`를 계산하기 전, 
    1. 모멘텀 방법으로 인해 이동될 방향을 미리 예측
    2. 해당 방향으로 얼마간 미리 이동한 뒤 gradient를 계산하는 효과
- 속도는 그대로이면서 이동 전에 한 단계를 미리 예측함으로써 불필요한 이동을 줄일 수 있음

### 구현

1. python

```python
v = m * v - learning_rate * gradient(self.weight[i-1] + m * v)
self.weight[i] += v 
```

2. Keras

```python
keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
```



## 4. 아다그라드

- Adagrad, Adaptive Gradient

$$
G(t)=G(t-1)+[\frac{\partial오차}{\partial W(t)}]^2\\W(t+1)=W(t)+\eta\frac{1}{\sqrt{Gt+\epsilon}}	
\cdot\frac{\partial오차}{\partial W(t)}
$$

- `G(t)`는 `t`스텝까지 각 스텝별로 변수가 수정된 gradient의 제곱을 모두 합한 값
- `ε`는 아주 작은 상수를 의미. 0으로 나누는 것을 방지해 줌

### 개념

- 변수의 업데이트 횟수에 따라 학습률을 조절하는 옵션이 추가됨
  - 자주 업데이트 된 변수(최적해에 가까워졌다고 가정)는 학습률을 줄여 세밀한 업데이트를 통해 예측 정확도를 높임
  - 업데이트가 충분히 이뤄지지 않은 변수는 학습률을 높여 예측 정확도를 높임
    1. 모멘텀 방법으로 인해 이동될 방향을 미리 예측
    2. 해당 방향으로 얼마간 미리 이동한 뒤 gradient를 계산하는 효과
- 속도는 그대로이면서 이동 전에 한 단계를 미리 예측함으로써 불필요한 이동을 줄일 수 있음

### 구현

1. python

```python
import numpy as np
g += gradient ** 2
self.weight[i] += learning_rate * gradient / (np.sqrt(g + e))
```

2. Keras

```python
keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6)
```



## 5. Rmsprop

$$
G(t)=\gamma G(t-1)+(1-\gamma)	
\cdot[\frac{\partial오차}{\partial W(t)}]^2\\W(t+1)=W(t)+\eta\frac{1}{\sqrt{Gt+\epsilon}}	
\cdot\frac{\partial오차}{\partial W(t)}
$$

### 개념

- 아다그라드의 `G(t)`값이 무한히 커지는 것을 방지
- 논문이 아닌 Coursera 수업에서 소개됨
- 이동 지수의 평균을 이용한 방법
- `γ`계수가 이전 값과 수정 값 사이의 적용 비율을 조절해 줌
  - 이전 기울기의 영향을 억제하는 효과
  - 아다그라드의 `G(t)`값에 해당하는 부분이 급격히 변하는 것을 방지하는 효과

### 구현

1. python

```python
import numpy as np
g = gamma * g (1 - gamma) * gradient ** 2
self.weight[i] += learning_rate * gradient / (np.sqrt(g + e))
```

2. Keras

```python
keras.optimizers.RMSprop(lr=0.001, rho=0.9 epsilon=1e-08, decay=0.0)
```



## 5. Adam

$$
V(t)=\gamma_1G(t-1)+(1-\gamma_1)	
\cdot\frac{\partial오차}{\partial W(t)}\\G(t)=\gamma_2 G(t-1)+(1-\gamma_2)	
\cdot[\frac{\partial오차}{\partial W(t)}]^2
$$

- 위 두 값을 다음과 같이 대입함

$$
\hat{V}(t)=\frac{V(t)}{1-\gamma^t_1},\;\;\hat{G}(t)=\frac{G(t)}{1-\gamma^t_2}\\W(t+1)=W(t)-\eta\cdot\frac{\hat{G}(t)}{\sqrt{\hat{V}(t)+\epsilon}}
$$

### 개념

- Rmsprop에서 한 단계 업그레이드된 방법
- `G(t)`값을 구하지만 모멘텀SGD와도 유사하게 gradient를 제곱하지 않는 `V(t)`값 또한 사용함
- Rmsprop과 모멘텀SGD의 장점을 취하는 방법

### 구현

1. python

```python
import numpy as np
v = gamma1 * m + (1 - gamma1) * dx
g = gamma2 * v + (1 - gamma2) * (dx ** 2)
x += learning_rate * g / (np.sqrt(v + e))
```

2. Keras

```python
keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
```

