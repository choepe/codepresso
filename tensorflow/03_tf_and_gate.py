import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 파라미터
input_dim = 2
hidden_units = 1
learning_rate = 0.01

# 가중치(input 특성: 2/ 퍼셉트론: 1)
w = tf.Variable(tf.random.uniform(shape=(input_dim, hidden_units)))

# 편향(퍼셉트론: 1)
b = tf.Variable(tf.zeros(shape=(hidden_units,)))

# 퍼셉트론의 수학 모델 f(x*w + b)
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
        gradient_lw, gradient_lb = tape.gradient(loss, [w, b])
    w.assign(w - learning_rate * gradient_lw)
    b.assign(b - learning_rate * gradient_lb)
    return loss

inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=np.float32)

labels = np.array([0, 0, 0, 1], dtype=np.float32)

# view
#plt.scatter(inputs[:, 0], inputs[:, 1], c=labels[:])
#plt.show()

for epoch in range(100):
    for x, y in zip(inputs, labels):
        loss = train([x], [y])
    print("Epoch {}: loss={}".format(epoch+1, float(loss)))

#predictions = predict(inputs)

#plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:]> 0.5)
#plt.show()

test_inputs = np.random.uniform(0, 1, (5000, 2)).astype(np.float32)
predictions = predict(test_inputs)

plt.scatter(test_inputs[:, 0], test_inputs[:, 1], c=predictions[:]> 0.5)
plt.show()

'''
for x, y in zip(inputs, labels):
    print(x, y)
    loss = train([x], [y])

predictions = predict(inputs)
loss = mse_loss(labels, predictions)
tf.square(labels - predictions)
inputs.shape
'''