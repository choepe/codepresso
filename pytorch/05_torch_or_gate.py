import torch
import matplotlib.pyplot as plt

# 파라미터 값(input 특성 개수, 퍼셉트론의 개수, 학습률) 설정
input_dim = 2
hidden_units = 1
learning_rate = 0.01

# 가중치(input 특성 : 2/ 퍼셉트론 : 1)
w = torch.rand(size=(input_dim, hidden_units), requires_grad=True)

# 편향(퍼셉트론 : 1)
b = torch.zeros(size=(hidden_units,), requires_grad=True)

# 퍼셉트론의 수학 모델 f(x*w + b) 구현
def predict(input):
  # x*w + b 연산 구현
  x = torch.matmul(input, w) + b
  # relu 활성화 함수 구현
  x = torch.maximum(torch.tensor(0.), x)
  return x

# loss(mse)
def mse_loss(labels, predictions):
  # MSE(Mean Square Error) 연산 구현
  loss = torch.mean(torch.square(labels - predictions))
  return loss

# train
def train(inputs, labels):
  # 퍼셉트론 모델을 예측값을 계산
  predictions = predict(inputs)
  # 모델의 예측값과 정답간의 에러를 loss 을 이용해 계산
  loss = mse_loss(labels, predictions)
  # 모델의 파라미터(w, b)가 loss 값에 미치는 영향도를 미분(오차역전파)을 통해 계산
  loss.backward()
  # 경사하강법을 수행하여 모델의 파라미터(w, b) 업데이트
  w.data = w - learning_rate*w.grad
  b.data = b - learning_rate*b.grad
  w.grad = None
  b.grad = None
  return loss

# 퍼셉트론 모델 학습을 위한 OR Gate 데이터 생성
inputs = torch.tensor([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]], dtype=torch.float32)

labels = torch.tensor([0, 1, 1, 1], dtype=torch.float32)

plt.scatter(inputs[:, 0], inputs[:, 1], c=labels[:])
plt.show()

# train 함수를 반복적으로 실행하여 퍼셉트론 모델을 학습
for epoch in range(100):
  # input 데이터와 label 데이터를 한 건씩 추출하여 train 함수에 전달
  for x, y in zip(inputs, labels):
    loss = train(inputs, labels)
  # 학습 중간에 loss 값의 변화 출력(코드 제출시 주석 처리)
  if (epoch+1)%10 == 0:
   print("Epoch {}: loss={}".format(epoch+1, float(loss)))

# 학습된 모델에 input 데이터 입력하여 예측결과 계산
predictions = predict(inputs)


print('*******모델의 예측 결과*******')
print(predictions[:]> 0.5)