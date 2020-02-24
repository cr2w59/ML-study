# 로지스틱 회귀

- Logistic Regression
- T/F를 판단하는 과정
  - 주어진 입력 값의 특징을 추출해 저장해 만든 model을 이용해 추후 비슷한 입력값에 대한 T/F를 판단

### 1. 정의

: 선형 회귀와 마찬가지로 적절한 선을 그리는 과정이지만 직선이 아니라 참(1)과 거짓(0) 사이를 구분하는 S자 형태의 선을 긋는 작업

| 공부한 시간 |   2    |   4    |   6    |  8   |  10  |  12  |  14  |
| :---------: | :----: | :----: | :----: | :--: | :--: | :--: | :--: |
|  합격 여부  | 불합격 | 불합격 | 불합격 | 합격 | 합격 | 합격 | 합격 |

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20200215170058217.png" alt="image-20200215170058217" style="zoom: 50%;" />



### 2. 시그모이드 함수

- Sigmoid Function

$$
y = \frac{1}{1+e^{(-ax+b)}}
$$

-  *e*: 자연 상수(2.71828...)

-  ***a*: 그래프의 경사도 (a와 경사도는 비례)**

    ​	<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20200215172412101.png" alt="image-20200215172412101" style="zoom: 67%;" />

    - **a가 작아질수록 오차는 무한대로 커지지만, a가 커진다고 해서 오차가 없어지지는 않음**

      <img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20200215173313016.png" alt="image-20200215173313016" style="zoom: 67%;" />

-  ***b*: 그래프의 좌우 이동**

    ​	<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20200215172536094.png" alt="image-20200215172536094" style="zoom: 67%;" />

    - **b가 너무 크거나 작을 경우 오차는 이차 함수 그래프와 유사한 형태로 나타남**

      ![image-20200215173348988](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20200215173348988.png)

### 3. 오차 공식

- 시그모이드 함수의 특징은 y값이 0과 1 사이라는 것
- 실제 값이 1일 때 예측 값이 0에 가까워지는 경우와 실제 값이 0일 때 예측 값이 1에 가까워지는 경우에 오차가 커짐 => 공식화한 것이 로그 함수

### 4. 로그 함수

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20200215174101895.png" alt="image-20200215174101895" style="zoom:50%;" />

- 파란선: 실제 값이 1일 때 `-log h`
- 빨간선: 실제 값이 0일 때 `-log(1-h)`

$$
loss = -\{y\log h+(1-y)\log(1-h)\}
$$

------

1. 리스트 x1_data, x2_data와 리스트 y_data를 만들어 데이터 저장, 학습률 설정

   ```python
   data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]]
   x_data = [row[0] for row in data]
   y_data = [row[1] for row in data]
   
   learning_rate = 0.5
   ```
   
2. 기울기 a와 y절편 b의 값을 임의의 정함

   ```python
   a = tf.Variable(tf.random_uniform([1], dtype=tf.float64, seed=0))
   b = tf.Variable(tf.random_uniform([1], dtype=tf.float64, seed=0))
   ```
   
3. y에 대한 시그모이드 식을 세움

   ```python
   y = 1 / (1 + np.e**(-a * x_data + b))
   ```
   
4. 텐서플로 함수들로 RMSE를 구현

   ```python
   loss = -tf.reduce_mean(np.array(y_data) * tf.log(y) + (1 - np.array(y_data)) * tf.log(1 - y))
   ```
   
5. loss가 최소인 값 찾기

   ```python
   gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
   ```

6. 결과 값 출력

   ```python
   with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     for i in range(60001):
       sess.run(gradient_descent)
       if i % 6000 == 0:
         print('Epoch: %.f, loss=%.04f, 기울기 a=%.4f, 바이어스 b=%.4f' %(i, sess.run(loss), sess.run(a), sess.run(b))) 
   ```

- logistic_regression.py

```python
import tensorflow as tf
import numpy as np

# x, y의 데이터 값
data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]]
x_data = [row[0] for row in data]
y_data = [row[1] for row in data]

# 학습률 값
learning_rate = 0.5

# a와 b의 값을 임의로 정함
a = tf.Variable(tf.random_uniform([1], dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_uniform([1], dtype=tf.float64, seed=0))

# y 시그모이드 함수의 방정식을 세움
y = 1 / (1 + np.e**(-a * x_data + b))

# loss를 구하는 함수
loss = -tf.reduce_mean(np.array(y_data) * tf.log(y) + (1 - np.array(y_data)) * tf.log(1 - y))

# loss를 최소로 하는 값 찾기
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 학습
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(60001):
    sess.run(gradient_descent)
    if i % 6000 == 0:
      print('Epoch: %.f, loss=%.04f, 기울기 a=%.4f, 바이어스 b=%.4f' %(i, sess.run(loss), sess.run(a), sess.run(b))) 
```

![image-20200217174044222](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20200217174044222.png)

#### 여러 입력 값을 갖는 로지스틱 회귀

------

1. 난수 고정

   ```python
   SEED = 0
   np.random.seed(SEED)
   tf.set_random_seed(SEED)
   ```

2. type을 맞추고, x, y 데이터 값을 저장, 학습률 지정

   ```python
   x_data = np.array([[2, 3], [4, 3], [6, 4], [8, 6], [10, 7], [12, 8], [14, 9]])
   y_data = np.array([0, 0, 0, 1, 1, 1, 1]).reshape(7, 1)
   
   learning_rate = 0.1
   ```

3. 플레이스 홀더를 만들고 a와 b의 값을 임의로 정함

   ```python
   X = tf.placeholder(tf.float64, shape=[None, 2])
   Y = tf.placeholder(tf.float64, shape=[None, 1])
   
   a = tf.Variable(tf.random_uniform([2, 1], dtype=tf.float64))
   b = tf.Variable(tf.random_uniform([1], dtype=tf.float64))
   ```

4. 텐서플로 내장함수를 이용해 y 시그모이드 함수의 방정식을 세움

   ```python
   y = tf.sigmoid(tf.matmul(X, a) + b)
   ```

5. 텐서플로 내장함수를 이용해 loss 변수를 만들고 loss가 최소인 값을 찾을 옵티마이저 만듦

   ```python
   loss = -tf.reduce_mean(Y * tf.log(y) + (1 - Y) * tf.log(1 - y))
   gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
   ```

6. 평가지수 변수 지정

   ```python
   predicted = tf.cast(y > 0.5, dtype=tf.float64)
   accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float64))
   ```
   
7. 학습 및 결과 값 출력

   ```python
   with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     for i in range(3001):
       a_, b_, loss_, _ = sess.run([a, b, loss, gradient_descent], 
                                   feed_dict={X:x_data, Y:y_data})
       if (i + 1) % 300 == 0:
         print('step: %.f, a1=%.4f, a2=%.4f, b=%.4f, loss=%.04f' %
               (i+1, a_[0], a_[1], b_, loss_))
   ```

- multi_logistic_regression.py

```python
import tensorflow as tf
import numpy as np

SEED = 0
np.random.seed(SEED)
tf.set_random_seed(SEED)

# x, y의 데이터 값
x_data = np.array([[2, 3], [4, 3], [6, 4], [8, 6], [10, 7], [12, 8], [14, 9]])
y_data = np.array([0, 0, 0, 1, 1, 1, 1]).reshape(7, 1)

# 학습률
learning_rate = 0.1

# 입력 값을 담을 플레이스 홀더 만듦
X = tf.placeholder(tf.float64, shape=[None, 2])
Y = tf.placeholder(tf.float64, shape=[None, 1])

# a와 b의 값을 임의로 정함
a = tf.Variable(tf.random_uniform([2, 1], dtype=tf.float64))
b = tf.Variable(tf.random_uniform([1], dtype=tf.float64))

# y 시그모이드 함수의 방정식을 세움
y = tf.sigmoid(tf.matmul(X, a) + b)

# loss를 구하는 함수
loss = -tf.reduce_mean(Y * tf.log(y) + (1 - Y) * tf.log(1 - y))

# loss를 최소로 하는 값 찾기
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

predicted = tf.cast(y > 0.5, dtype=tf.float64)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float64))

# 학습
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(3001):
    a_, b_, loss_, _ = sess.run([a, b, loss, gradient_descent], 
                                feed_dict={X:x_data, Y:y_data})
    if (i + 1) % 300 == 0:
      print('step: %.f, a1=%.4f, a2=%.4f, b=%.4f, loss=%.04f' %
            (i+1, a_[0], a_[1], b_, loss_))
```

##### 실제 값 적용

```python
# 학습
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(3001):
    a_, b_, loss_, _ = sess.run([a, b, loss, gradient_descent], 
                                feed_dict={X:x_data, Y:y_data})
    if (i + 1) % 300 == 0:
      print('step: %.f, a1=%.4f, a2=%.4f, b=%.4f, loss=%.04f' %
            (i+1, a_[0], a_[1], b_, loss_))
  # 적용
  new_x = np.array([7, 6]).reshape(1, 2)
  new_y = sess.run(y, feed_dict={X:new_x})
  print(f"공부한 시간: %d, 과외 수업 횟수: %d" % (new_x[:,0], new_x[:,1]))
  print(f"합격 가능성: %5.2f%%" %(new_y*100))
```

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20200219094749666.png" alt="image-20200219094749666" style="zoom:67%;" />

