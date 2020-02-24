# 퍼셉트론(Perceptron)

<img src="https://github.com/cr2w59/pengsoo/blob/master/dl/doc/images/%EC%A3%BC%EC%84%9D%202020-02-20%20130252.png?raw=true" style="zoom:110%;" />

## 개념

$$
y = wx + b \\ (w:weight,\;x:input,\;b:bias)
$$



1. **가중합(weighted sum)**: 입력 값(`x`)과 가중치(`w`)의 곱을 모두 더한 후 바이어스(`b`)를 더한 값

   - 가중합의 결과를 놓고 1 또는 0를 출력해 다음 단계로 보냄

2. **활성화 함수(activation function)**: 가중합의 결과를 보고 1 또는 0으로 판단하는 함수

   

## XOR 문제

![](https://github.com/cr2w59/pengsoo/blob/master/dl/doc/images/%EC%A3%BC%EC%84%9D%202020-02-20%20125410.png?raw=true)

- AND와 OR 게이트는 직선을 그어 1을 구분할 수 있지만 XOR의 경우에는 불가능함

- '뉴런 -> 신경망 -> 지능'이라는 도식 그대로 '퍼셉트론 -> 인공 신경망 -> 인공지능'이 불가능해짐

- XOR문제를 해결한 개념이 **다층 퍼셉트론(Multilayer Perceptron)** 
  
  <img src="https://github.com/cr2w59/pengsoo/blob/master/dl/doc/images/%EC%A3%BC%EC%84%9D%202020-02-20%20125411.png?raw=true" style="zoom:50%;" />
  
  - 2차 좌표 평면 자체에 변화를 주면 해결 가능 => **은닉층(Hidden Layer)**
  
    
  
    <img src="http://colah.github.io/posts/2015-09-NN-Types-FP/img/netvis.png" style="zoom: 100%;" />
  
  - 입력값을 놓고 파란색과 빨간색의 영역을 단순 직선으로 구분할 수 없음
  
  - 은닉층을 만들어 공간을 왜곡하면 두 영역을 가로지르는 선이 직선으로 바뀜
  
  - https://goo.gl/8qEGHD



# 다층 퍼셉트론(Multilayer Perceptron)

## 개념

- 퍼셉트론 하나로는 해결되지 않던 문제를 은닉층을 만들어 해결
- 은닉층을 여러 개 쌓아 문제를 해결하는 것 => **신경망** 

## 설계

![](https://github.com/cr2w59/pengsoo/blob/master/dl/doc/images/%EC%A3%BC%EC%84%9D%202020-02-20%20125412.png?raw=true)

1. `x`: 입력

2. `w`: 가중치,  `b`: bias

   - 여기서는 은닉층을 포함해 가중치 6개와 바이어스 3개가 필요함
     $$
     W(1)= \begin{bmatrix}w_{11} & w_{12} \\\\w_{21} & w_{22} \end{bmatrix}\;B(1)= \begin{bmatrix}b_1\\\\b_2 \end{bmatrix}\\
     W(2)= \begin{bmatrix}w_{31} \\\\w_{32} \end{bmatrix}\;B(2)= \begin{bmatrix}b_3\end{bmatrix}
     $$

3. `n`: node

$$
n1 = \sigma(x_1w_{11}+x_2w_{21}+b_1)\\ n2 = \sigma(x_1w_{12}+x_2w_{22}+b_2)
$$

- Sigmoid 
  $$
  y = \frac{1}{1+e^{(-ax+b)}}
  $$
  

4. `y`: 출력
   - 출력층에서도 시그모이드 함수로 y값이 정해짐

$$
y_{out} = \sigma(n_1w_{31}+n_2w_{32}+b_3)
$$

- 각각의 퍼셉트론이 가운데 숨어있는 은닉층으로  자신의 `w`와 `b`값을 보내고, 은닉층에서 모은 값을 시그모이드 함수를 이용해 최종 값으로 결과를 보냄
- *NAND* 게이트와 *OR* 게이트 node를 지난 값에 대해 *AND* 게이트를 수행한 값이 `Y`

------

1. `w`와 `b`선언

   ```python
   import numpy as np
   
   w11 = np.array([-2, -2])
   w12 = np.array([2, 2])
   w2 = np.array([1, 1])
   b1 = 3
   b2 = -1
   b3 = -1
   ```

2. 퍼셉트론 함수 구현

   ```python
   def MLP(x, w, b):
       y = np.sum(w * x) + b
       if y <= 0:
           return 0
       else:
           return 1
   ```

3. 각 게이트 함수 구현

   ```python
   # NAND GATE
   def NAND(x1, x2):
       return MLP(np.array([x1, x2]), w11, b1)
   
   # OR GATE
   def OR(x1, x2):
       return MLP(np.array([x1, x2]), w12, b2)
   
   # AND GATE
   def AND(x1, x2):
       return MLP(np.array([x1, x2]), w2, b3)
   
   # XOR GATE
   def XOR(x1, x2):
       return AND(NAND(x1, x2), OR(x1, x2))
   ```

4. x₁과 x₂값을 번갈아 대입해 가며 최종 값 출력

   ```python
   if __name__=="__main__":
       for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
           y = XOR(x[0], x[1])
           print('입력값: ' + str(x) + '출력값:' + str(y))
   ```

- xor.py

```python
import numpy as np

# 가중치와 바이어스
w11 = np.array([-2, -2])
w12 = np.array([2, 2])
w2 = np.array([1, 1])
b1 = 3
b2 = -1
b3 = -1

# 퍼셉트론
def MLP(x, w, b):
    y = np.sum(w * x) + b
    if y <= 0:
        return 0
    else:
        return 1

# NAND GATE
def NAND(x1, x2):
    return MLP(np.array([x1, x2]), w11, b1)

# OR GATE
def OR(x1, x2):
    return MLP(np.array([x1, x2]), w12, b2)

# AND GATE
def AND(x1, x2):
    return MLP(np.array([x1, x2]), w2, b3)

# XOR GATE
def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))

# x1, x2 값을 번갈아 대입해 가며 최종값 출력
if __name__=="__main__":
    for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        y = XOR(x[0], x[1])
        print('입력값: ' + str(x) + ' 출력값: ' + str(y))
```

![](https://github.com/cr2w59/pengsoo/blob/master/dl/doc/images/xor_output.png?raw=true)


