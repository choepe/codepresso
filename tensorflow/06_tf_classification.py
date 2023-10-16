import tensorflow as tf
from keras.datasets import fashion_mnist
from keras import models, layers

import numpy as np
import matplotlib.pyplot as plt

# 데이터셋 임포트, 데이터 탐색
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images.shape
train_labels.shape
test_images.shape
test_labels.shape

# 데이터의 전처리
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#train_images = train_images.reshape(60000, 28*28) / 255.0
#test_images = test_images.reshape(10000, 28*28) / 255.0
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()



# 모델 구성
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 모델 훈련
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Grab an image from the test dataset.
img = test_images[1]
img.shape

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
img.shape

predictions_single = probability_model.predict(img)
predictions_single

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

np.argmax(predictions_single[0])

# 수강생 작성 코드
# 1. 1차원 형태의(batch, ) class index 인 train, test label 데이터를
#    to_categorical API를 이용하여 one-hot-encoding 수행하여 2차원(batch, class_cnt) 으로 변경 하세요



"""* Step 3. DNN(MLP) 모델 디자인"""

# 수강생 작성 코드
# 1. Sequential API를 이용하여 fashion_mnist 데이터 셋을 분석 하기 위한 MLP 모델을 디자인 하세요



"""* Step 4. 모델의 학습 정보 설정"""

# 수강생 작성 코드
# 1. tf.keras.Model 객체의 compile 메서드를 이용하여 학습을 위한 정보들을 설정하세요
#   - optimizer
#   - loss : categorical_crossentropy 로 설장(label 데이터를 one-hot-encoding 하였기 때문)
#   - metrics : 체점 기준인 accuracy 로 설정



"""* Step 5. 모델에 input, target 데이터 연결 후 학습"""

# 수강생 작성 코드
# 1. tf.keras.Model 객체의 fit 메서드를 이용하여 모델을 학습하세요
#   - fit 메서드의 verbose=2 로 설정 하세요



"""* 모델 제출 """

# 학습된 모델을 제출하기 위한 코드 입니다. 수정하지 마세요
model.save('my_model.h5')