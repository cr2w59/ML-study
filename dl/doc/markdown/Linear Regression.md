# 선형 회귀

- Linear Regression

## 최소 제곱법

- method of least squares

-  `y = ax + b`의 기울기 a와 y절편 b를 구할 수 있다.


$$
  a = {\frac{\sum_{=1}^N(x_i-mean(x))(y_i-mean(y))}{\sum_{i=1}^N(x_i-mean(x))^2}}\\ = {\frac{(x-x평균))(y-y평균)의 합}{(x-x평균)^2의 합}}
$$

$$
b = {mean(y) - (mean(x)*a)}\\ = {y의 평균-(x의 평균*기울기a)}
$$

------

1. 리스트 x와 리스트 y를 만들어 데이터 저장하고 평균값을 구해놓음

   ```python
   x = [2, 4, 6, 8]
   y = [81, 93, 91, 97]
   mx = np.mean(x)
   my = np.mean(y)
   ```

2. 최소 제곱근 공식 중 **분모의 값**, 즉 'x의 각 원소와 x의 평균값들의 차를 제곱'한 값을 구해 divisor 변수에 저장

   ```python
   divisor = sum([mx-i]**2 for i in x)
   ```

3. **분자의 값**을 구하는 함수 top를 구현

   ```python
   def top(x, mx, y, my):
     tmp = 0
     for i in range(len(x)):
       tmp += (x[i]-mx) * (y[i]-my)
     return tmp
   dividend = top(x, mx, y, my)
   ```

4. **기울기**와 **y절편**(기울기를 이용해서)을 구함 

   - *a*: 기울기 값
   - *b*: y절편 값

   ```python
   a = dividend / divisor
   b = my - (mx * a)
   
   print(f'기울기 a={a}')
   print(f'y 절편 b={b}')
   ```


- mls.py

```python
import numpy as np

# x값과 y값
x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

# x과 y의 평균값
mx = np.mean(x)
my = np.mean(y)

# 기울기 공식의 분모
divisor = sum([mx-i]**2 for i in x)

# 기울기 공식의 분자
def top(x, mx, y, my):
  d = 0
  for i in range(len(x)):
    d += (x[i]-mx) * (y[i]-my)
  return d

dividend = top(x, mx, y, my)

# 기울기와 y절편 구하기
a = dividend / divisor
b = my - (mx*a)

# 출력
print(f'기울기 a={a}')
print(f'y 절편 b={b}')
```



## 평균 제곱근 오차(RMSE)

### 1. 평균 제곱 오차(MSE)

- mean square error
  $$
  MSE = {\frac{1}{N}\sum_{i=1}^N(y_i-p_i)^2}
  $$
  



### 2. 평균 제곱근 오차(RMSE)

$$
RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^N(y_i-p_i)^2}
$$

- root mean square error
- **오차 평가 알고리즘**
- MSE는 대용량 데이터를 이용할 때 계산 속도가 느려질 수 있다. 그래서 여기에 제곱근을 씌워 준다.
- 선형 회귀란 임의의 직선에 대한 RMSE 값을 가장 작게 만들어 주는 a와 b를 구하는 작업

------

1. 임의로 정한 기울기 a와 y절편 b의 값이 각각 3과 76이라고 할 때 리스트 ab를 만들어 저장

   ```python
   ab = [3, 76]
   ```

2. 리스트 x와 리스트 y를 만들어 첫 번째 값을 리스트 x에 저장하고 두 번째 값을 리스트에 저장

   ```python
   data = [[2,81], [4,93], [6,91], [8,97]]
   x = [i[0] for i in data]
   y = [i[1] for i in data]
   ```

3. 일차 방정식 `y=ax+b`를 구현

   ```python
   def predict(x):
       return ab[0]*x + ab[1]
   ```

4. RMSE 공식 함수로 구현

   - *p*: 예측 값
   - *a*: 실제 값

   ```python
   def rmse(p, a):
   	return np.sqrt(((p-a)**2).mean())
   ```

5. rmse()에 데이터를 대입하여 최종값을 구하는 함수 구현

   - *predict_result*: predict()의 return 값이 들어감

   ```python
   def rmse_val(predict_result, y):
     return rmse(np.array(predict_result), np.array(y))
   ```

6. 예측 값을 담는 리스트 predict_result 구현

   ```python
   predict_result = list()
   
   for i in range(len(x)):
     predict_result.append(predict(x[i]))
     print(f'{x[i]}, {y[i]}, {predict(x[i])}')
   ```


- rmse.py

```python
import numpy as np

# 기울기 a와 y절편 b
ab = [3, 76]

# x, y의 데이터 값
data = [[2,81], [4,93], [6,91], [8,97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# y=ax+b에 a와 b 값을 대입하여 결과를 출력하는 함수
def predict(x):
  return ab[0]*x + ab[1]

# RMSE 함수
def rmse(p, a):
  return np.sqrt(((p-a)**2).mean())

# RMSE 함수를 각 y값에 대입해 최종값을 구하는 함수
def rmse_val(predict_result, y):
  return rmse(np.array(predict_result), np.array(y))

# 예측 값이 들어갈 빈 리스트
predict_result = list()

# 모든 x값을 한 번씩 대입하여 predict_result 리스트를 완성
for i in range(len(x)):
  predict_result.append(predict(x[i]))
  print(f'{x[i]}, {y[i]}, {predict(x[i])}')

# 최종 RMSE 출력
print(f'rmse 최종값: ' + str(rmse_val(predict_result, y)))

```



## 경사 하강법

- gradient descent
- `y = x²`의 그래프에서 오차를 비교해 가장 작은 방향으로 이동시키는 방법
- 미분 기울기를 이용함

### 1. 미분의 개념

![](https://github.com/cr2w59/pengsoo/blob/master/dl/doc/images/image-20200215142225800.png?raw=true)

- x값이 아주 미세하게 움직일 때의 y 변화량을 구한 뒤, 이를 x의 변화량으로 나누는 과정
  * **순간 변화율**: a가 변화량이 0에 가까울 만큼 아주 미세하게 변화한 방향성
  * **기울기**: 순간 변화율의 방향에 맞추어 그은 접선


$$
\frac{d}{dx}f(x) = {lim_{Δx\rightarrow 0}}\frac{f(x+Δx)-f(x)}{Δx}
$$

### 2. 경사 하강법의 개요

*최솟값에서의 기울기는 x축과 평행한 선이 됨. 즉, 기울기가 0*

**미분 값이 0인 지점**을 찾는 과정

1. a₁에서 미분을 구함
2. 구해진 기울기의 반대 방향(기울기가 +면 음의 방향, -면 양의 방향)으로 얼마간 이동시킨 a₂에서 미분을 구함
3. a₃에서 미분을 구함
4. 3.의 값이 0이 아니면 위 과정을 반복해서 기울기가 0인 한 점으로 수렴하도록 함

### 3. 학습률

기울기의 부호를 바꿔 이동시킬 때 적절한 거리를 찾지 못해 너무 멀리 이동시키면 a의 값이 한 점으로 모이지 않고 위로 치솟아 버림. **이동 거리를 정해주는 것이 학습률**

------

1. 리스트 x_data와 리스트 y_data를 만들어 데이터 저장, 학습률 설정

   ```python
   data = [[2, 81], [4, 93], [6, 91], [8, 97]]
   x_data = [row[0] for row in data]
   y_data = [row[1] for row in data]
   
   learning_rate = 0.1
   ```

2. 기울기 a와 y절편 b의 값을 임의의 정함

   - 기울기의 범위는 0~10 사이이며, y절편은 0~100 사이에서 변하게 함
   - *random_uniform()*은 임의의 수를 생성
     - 뽑아낼 값의 개수 
     - 최솟값 및 최댓값
     - 데이터 형식
     - 실행 시마다 같은 값이 나올 수 있게 seed 고정

   ```python
   a = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
   b = tf.Variable(tf.random_uniform([1], 0, 100, dtype=tf.float64, seed=0))
   ```

3. y에 대한 일차 방정식 `ax+b`의 식을 세움

   ```python
   y = a * x_data + b
   ```

4. 텐서플로 함수들로 RMSE를 구현

   - *tf.sqrt(x)* : x의 제곱근 계산

   - *tf.reduce_mean(x)* : x의 평균 계산

   - *tf.square(x)* : x의 제곱 계산
     $$
     RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^N(y_i-p_i)^2}
     $$
     

   ```python
   rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))
   ```

5. 경사 하강법을 실행하는 단계

   - *tf.train.GradientDescentOptimizer()*: 텐서플로 경사 하강법 함수
     - *learning_rate*: 학습률
     - 평균 제곱근 오차가 최소가 되는 지점을 경사 하강법으로 찾아라
   - *gradient_descent*: 오차가 제일 작은 지점(경사 하강법의 실행 결과)

   ```python
   gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)
   ```

6. 결과 값 출력

   - 텐서플로에서 실행은 모두 Session의 역할
   - Session을 통해 구현될 함수를 **그래프**라고 부름
     - Session이 할당되면 session.run('그래프명')의 형식으로 해당 함수를 구동
   - *tf.global_variables_initializer()* : 변수 초기화 함수

   ```python
   with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     for step in range(2001):
       sess.run(gradient_descent)
       if step % 100 == 0:
         print('Epoch: %.f, RMSE = %.04f, 기울기 a = %.4f, y절편 b = %.4f' %(step, sess.run(rmse), sess.run(a), sess.run(b))) 
   ```

   * *Epoch*: 입력 값에 대해 몇 번이나 반복해 실험했는지를 나타냄

- gradient_descent.py

```python
import tensorflow as tf

# x, y의 데이터 값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x_data = [x_row[0] for x_row in data]
y_data = [y_row[1] for y_row in data]

# 학습률 값
learning_rate = 0.1

# 기울기 a와 y절편 b의 값을 임의로 정함
a = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype=tf.float64, seed=0))

# y에 대한 일차 방정식 ax+b의 식을 세움
y = a * x_data + b

# 텐서플로 RMSE 함수
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

# RMSE 값을 최소로 하는 값 찾기
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

# 텐서플로를 이용한 학습
with tf.Session() as sess:
  # 변수 초기화
  sess.run(tf.global_variables_initializer())
  # 2001번 실행(0번째를 포함하므로)
  for step in range(2001):
    sess.run(gradient_descent)
    # 100번마다 결과 출력
    if step % 100 == 0:
      print('Epoch: %.f, RMSE = %.04f, 기울기 a = %.4f, y절편 b = %.4f' %(step, sess.run(rmse), sess.run(a), sess.run(b)))
```

![image-20200215155339729](https://github.com/cr2w59/pengsoo/blob/master/dl/doc/images/image-20200215155339729.png?raw=true)



# 다중 선형 회귀

- Multi Linear Regression
- 여러 개의 독립 변수가 존재함
- `y = a₁x₁ + a₂x₂ + b`의 기울기 a₁, a₂와 y절편 b를 구해야 함

------

1. 리스트 x1_data, x2_data와 리스트 y_data를 만들어 데이터 저장, 학습률 설정

   ```python
   data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
   x1_data = [row[0] for row in data]
   x2_data = [row[1] for row in data]
   y_data = [row[2] for row in data]
   
   learning_rate = 0.1
   ```

2. 기울기 a와 y절편 b의 값을 임의의 정함

   ```python
   a1 = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
   a2 = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
   b = tf.Variable(tf.random_uniform([1], 0, 100, dtype=tf.float64, seed=0))
   ```

3. y에 대한 이차 방정식 `a₁x₁ + a₂x₂ + b`의 식을 세움

   ```python
   y = a1 * x1_data + a2 * x2_data + b
   ```

4. 텐서플로 함수들로 RMSE를 구현

   ```python
   rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))
   ```

5. 경사 하강법을 실행하는 단계

   ```python
   gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)
   ```

6. 결과 값 출력

   ```python
   with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     for step in range(2001):
       sess.run(gradient_descent)
       if step % 100 == 0:
         print('Epoch: %.f, RMSE=%.04f, 기울기 a1=%.4f, 기울기 a2=%.4f, y절편 b=%.4f' %(step, sess.run(rmse), sess.run(a1), sess.run(a2), sess.run(b))) 
   ```

- gradient_descent.py

```python
# x1, x2, y의 데이터 값
data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1_data = [row[0] for row in data]
x2_data = [row[1] for row in data]
y_data = [row[2] for row in data]

# 학습률 값
learning_rate = 0.1

# 기울기 a와 y절편 b의 값을 임의로 정함
# 단, 기울기의 범위는 0~10 사이이며, y절편은 0~100 사이에서 변하게 함
a1 = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
a2 = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype=tf.float64, seed=0))

# y에 대한 이차 방정식 a1x1 + a2x2 + b의 식을 세움
y = a1 * x1_data + a2 * x2_data + b

# 텐서플로 RMSE 함수
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

# RMSE 값을 최소로 하는 값 찾기
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

# 텐서플로를 이용한 학습
with tf.Session() as sess:
  # 변수 초기화
  sess.run(tf.global_variables_initializer())
  # 2001번 실행(0번째를 포함하므로)
  for step in range(2001):
    sess.run(gradient_descent)
    # 100번마다 결과 출력
    if step % 100 == 0:
      print('Epoch: %.f, RMSE=%.04f, 기울기 a1=%.4f, 기울기 a2=%.4f, y절편 b=%.4f' %(step, sess.run(rmse), sess.run(a1), sess.run(a2), sess.run(b))) 
```

![image-20200215161754724](https://github.com/cr2w59/pengsoo/blob/master/dl/doc/images/image-20200215161754724.png?raw=true)

#### 다중 선형 회귀를 그래프로 표현

- 1차원 예측 직선을 3차원 '예측 평면'으로 표현
- 새로운 독립 변수가 추가되면서 1차원 직선에서만 움직이던 예측 결과가 더 넓은 평면 범위 안에서 움직이게 되고, 좀 더 정밀한 예측을 할 수 있게 됨

![](https://github.com/cr2w59/pengsoo/blob/master/dl/doc/images/%EB%8B%A4%EC%A4%91%EC%84%A0%ED%98%95%ED%9A%8C%EA%B7%80.png?raw=true)

