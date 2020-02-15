import tensorflow as tf

# x1, x2, y의 데이터 값
data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1_data = [row[0] for row in data]
x2_data = [row[1] for row in data]
y_data = [row[2] for row in data]

# 기울기 a와 y절편 b의 값을 임의로 정함
# 단, 기울기의 범위는 0~10 사이이며, y절편은 0~100 사이에서 변하게 함
a1 = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
a2 = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype=tf.float64, seed=0))

# y에 대한 이차 방정식 a1x1 + a2x2 + b의 식을 세움
y = a1 * x1_data + a2 * x2_data + b

# 텐서플로 RMSE 함수
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

# 학습률 값
learning_rate = 0.1

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