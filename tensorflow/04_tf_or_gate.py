import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 파라미터 값(input 특성 개수, 퍼셉트론의 개수, 학습률) 설정
input_dim = 2
hidden_units = 1
learning_rate = 0.01

# 가중치(input 특성 : 2/ 퍼셉트론 : 1)
w = tf.Variable(tf.random.uniform(shape=(input_dim, hidden_units)))

# 편향(퍼셉트론 : 1)
b = tf.Variable(tf.zeros(shape=(hidden_units,)))

# 퍼셉트론의 수학 모델 f(x*w + b) 구현 f: relu
def predict(input):
  x = tf.matmul(input, w) + b
  x = tf.maximum(0, x)
  return x

# loss(mse)
def mse_loss(labels, predictions):
  loss = tf.reduce_mean(tf.square(labels - predictions))
  return loss

# train
def train(inputs, labels):
  with tf.GradientTape() as tape:
    predictions = predict(inputs)
    loss = mse_loss(labels, predictions)
    # 모델의 파라미터(w, b)가 loss 값에 미치는 영향도를 미분(오차역전파)을 통해 계산
    gradient_lw, gradient_lb = tape.gradients(loss, [w, b])
  # 경사하강법을 수행하여 모델의 파라미터(w, b) 업데이트
  w.assign(w - learning_rate * gradient_lw)
  b.assign(b - learning_rate * gradient_lb)
  return loss

# 퍼셉트론 모델 학습을 위한 OR Gate 데이터 생성
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=np.float32)

labels = np.array([0, 1, 1, 1], dtype=np.float32)

# OR Gate 데이터를 차트로 확인(코드 제출시 주석 처리)
plt.scatter(inputs[:, 0], inputs[:, 1], c=labels[:])
plt.show()

# train 함수를 반복적으로 실행하여 퍼셉트론 모델을 학습
for epoch in range(100):
  # input 데이터와 label 데이터를 한 건씩 추출하여 train 함수에 전달
  for x, y in zip(inputs, labels):
    loss = train(inputs, labels)
  if (epoch+1)%10 == 0:
    print("Epoch {}: loss={}".format(epoch+1, float(loss)))

# 학습된 모델에 input 데이터 입력하여 예측결과 계산
predictions = predict(inputs)

# 모델의 예측결과를 차트로 확인(코드 제출시 주석 처리)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:]> 0.5)
plt.show()

test_inputs = np.random.uniform(0, 1, (5000, 2)).astype(np.float32)
predictions = predict(test_inputs)

plt.scatter(test_inputs[:, 0], test_inputs[:, 1], c=predictions[:]> 0.5)
plt.show()


print('*******모델의 예측 결과*******')
print(predictions[:]> 0.5)