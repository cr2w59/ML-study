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
