import torch
import numpy as np
import matplotlib.pyplot as plt

# AND Gate
# [input_x1, input_x2] => [y]
# [0, 0] => [0]
# [0, 1] => [0]
# [1, 0] => [0]
# [1, 1] => [1]

# 파라미터(parameter) 값(input 특성 개수, 퍼셉트론의 개수, 학습률) 설정
input_dim = 2
units = 1
learning_rate = 0.001
# 가중치(weight)(input 특성: 2/ 퍼셉트론: 1)
w = torch.rand(size=(input_dim, units), requires_grad=True)
# 편향(bias)(퍼셉트론: 1)
b = torch.zeros(size=(units,), requires_grad=True)

w
w.shape
b
b.shape

input = torch.tensor([[0., 0.]])
input.shape
label = torch.tensor([0.])
label.shape
torch.maximum(input=torch.tensor(0, ), other=torch.tensor(1, ))

# 퍼셉트론(perceptron)의 수학 모델 f(x*w + b)
# activation function : f(x)는 relu
x = torch.matmul(input, w) + b
result = torch.maximum(input=torch.tensor(0.), other=x)

# loss(mse)
loss = torch.mean(torch.square(label - result))
loss.backward()
w.grad
b.grad


input = torch.tensor([[0., 1.]])
label = torch.tensor([0.])

x = torch.matmul(input, w) + b
result = torch.maximum(input=torch.tensor(0.), other=x)
loss = torch.mean(torch.square(label - result))
loss.backward()
w.grad
b.grad

w.data = w - learning_rate*w.grad
b.data = b - learning_rate*b.grad

w.grad = None
b.grad = None

def predict(input):
    x = torch.matmul(input, w) + b
    x = torch.maximum(input=torch.tensor(0.), other=x)
    return x

def mse_loss(labels, predictions):
    loss = torch.mean(torch.square(labels - predictions))
    return loss

def train(inputs, labels):
    predictions = predict(inputs)
    loss = mse_loss(labels, predictions)
    loss.backward()
    w.data = w - learning_rate*w.grad
    b.data = b - learning_rate*b.grad
    w.grad = None
    b.grad = None
    return loss

# [0, 0] => [0]
# [0, 1] => [0]
# [1, 0] => [0]
# [1, 1] => [1]
inputs = torch.tensor([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]], dtype=torch.float32)
labels = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
plt.scatter(inputs[:, 0], inputs[:, 1], c=labels[:]>0.5)
plt.show()

for epoch in range(100):
    for x, y in zip(inputs, labels):
        loss = train(x, y)
    print(f"Epoch {epoch + 1}: loss={float(loss)} ")

predictions = predict(inputs)


test_inputs = np.random.uniform(low=0, high=1, size=(5000, 2)).astype(np.float32)
test_inputs = torch.tensor(test_inputs)
test_predictions = predict(test_inputs)
test_predictions
plt.scatter(test_inputs[:, 0], test_inputs[:, 1], c=test_predictions[:]>0.5)
plt.show()