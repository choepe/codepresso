"""## DNN(MLP) 모델을 이용한 MNIST 데이터 셋 분류
* 체점 기준 :
  - 데이터 셋 : 체점 서버내 테스트 데이터 셋
  - 성능 지표 : Accuracy
  - PASS 기준 : 80% 이상
"""

import tensorflow as tf
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

"""* Step 1. Inptu tensor 와 Target tensor 준비(훈련데이터)"""

# label 데이터의 각 value 에 해당하는 class name 정보
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 수강생 작성 코드
# 1. import 한 fashion_mnist API를 이용하여 fashion_mnist 데이터 셋을 다운로드 받으세요



"""* Step 2. 데이터의 전처리 """

# 수강생 작성 코드
# 1. 3차원 형태(batch, hight, width)의 train, test feature 데이터를 2차원(batch, hight*width)으로 변경 하세요
# 2. feature 데이터를 [0, 1] 사이로 scailing을 수행하세요


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