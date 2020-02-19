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
    a_, b_, loss_, _ = sess.run([a, b, loss, gradient_descent], feed_dict={X:x_data, Y:y_data})
    if (i + 1) % 300 == 0:
      print('step: %.f, a1=%.4f, a2=%.4f, b=%.4f, loss=%.04f' %(i+1, a_[0], a_[1], b_, loss_))
  # 적용
  new_x = np.array([7, 6]).reshape(1, 2)
  new_y = sess.run(y, feed_dict={X:new_x})
  print(f"공부한 시간: %d, 과외 수업 횟수: %d" % (new_x[:,0], new_x[:,1]))
  print(f"합격 가능성: %5.2f%%" %(new_y*100))