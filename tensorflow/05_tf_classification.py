import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import models, layers

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images.shape
train_labels.shape

train_images[0].shape
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_labels[0]

# 전처리
train_images = train_images.reshape((60000, 28*28))
train_images.shape
test_images.shape
test_images = test_images.reshape((10000, 28*28))

model = models.Sequential()
model.add(layers.Dense(units=256, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(units=10, activation='softmax'))
model.summary()

w_cnt = 28*28*256
b_cnt = 256

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=train_images, y=train_labels, epochs=30, batch_size=128, validation_split=0.2)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(x=test_images, y=test_labels)

predict = model.predict(test_images[0].reshape((1, 28*28)))
np.argmax(predict[0])

plt.figure()
plt.imshow(test_images[0].reshape(28, 28))
plt.colorbar()
plt.grid(False)
plt.show()