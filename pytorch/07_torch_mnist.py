"""## DNN(MLP) 모델을 이용한 FASHION-MNIST 데이터 셋 분류
* 체점 기준 :
  - 데이터 셋 : 체점 서버내 테스트 데이터 셋
  - 성능 지표 : Accuracy
  - PASS 기준 : 80% 이상
"""

import torch
from keras.datasets import fashion_mnist
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
"""* Step 1. Inptu tensor 와 Target tensor 준비(훈련데이터)"""

# label 데이터의 각 value 에 해당하는 class name 정보
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 수강생 작성 코드
# 1. import 한 fashion_mnist API를 이용하여 fashion_mnist 데이터 셋을 다운로드 받으세요
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



"""* Step 2. 데이터의 전처리"""

# 수강생 작성 코드
# 1. 3차원 형태(batch, hight, width)의 train, test feature 데이터를 2차원(batch, hight*width)으로 변경 하세요
# 2. feature 데이터를 [0, 1] 사이로 scailing을 수행하세요



"""* Step 3. batch 구성을 위한 데이터 pipeline 생성"""

# 수강생 작성 코드
# 1. 훈련 데이터, 테스트 데이터를 파이토치의 텐서 형태로 변환해주세요


# 수강생 작성 코드
# 1. torch.utils.data.TensorDataset API를 활용해 훈련, 테스트 데이터를 위한 Dataset 객체를 생성 하세요


# 수강생 작성 코드
# 1. torch.utils.data.DataLoader API를 활용해 훈련, 테스트 데이터 pipeline을 생성하세요


"""* Step 4. DNN(MLP) 모델 디자인"""

# 수강생 작성 코드
# 1. torch.nn.Sequential API를 이용하여 fashion_mnist 데이터 셋을 분석 하기 위한 MLP 모델을 디자인 하세요



"""* Step 5. 모델의 학습 정보 설정"""

# 수강생 작성 코드
# 1. torch.nn 모듈에 구현된 API를 이용해 분류를 위한 손실함수 객체를 생성하세요
# 2. torch.optim 모듈에 구현된 API로 옵티마이저를 설정하세요



"""* Step 6. 모델의 학습 loop 를 정의하는 함수 생성"""

def train(epoch):
  # 수강생 작성 코드
  # 1. 모델 객체를 학습에 적합한 상태로 변경하세요


  # accuracy 계산을 위한 'correct' 변수 선언 및 0 으로 초기화
  correct = 0

  # 수강생 작성 코드
  # 1. 학습데이터 에서 미니 배치를 추출하여 학습 loop를 수행하는 for문을 작성 하세요
  for batch_idx, (____, ____) in enumerate(____):
    # 수강생 작성 코드
    # 1. 모델에 input data를 전달하여 순전파를 수행하세요


    # 수강생 작성 코드
    # 1. loss 함수에 모델의 예측값 과 정답 정보를 전달하여 loss 값을 계산 하세요


    # 수강생 작성 코드
    # 1. optimizer 객체를 이용해 모델을 구성하는 파라미터들(w, b)의 gradient를 초기화 하세요


    # 수강생 작성 코드
    # 1. 모델을 구성하는 파라미터들(w, b)이 loss 에 미치는 영향도(gradient)를 계산 하세요


    # 수강생 작성 코드
    # 1. 계산된 영향도(gradient) 값을 이용하여 모델의 파라미터(w,b)를 업데이트 하세요


    # 모델의 성능 지표를 계산 및 출력 하여 학습 진행을 체크
    correct += (output.argmax(dim=1) == target).type(torch.float).sum().item()

    if batch_idx%50==0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
            epoch,
            (batch_idx+1) * len(data),
            len(train_loader.dataset),
            100. * (batch_idx+1) / len(train_loader),
            batch_loss.item(),
            100. * correct / ((batch_idx+1) * len(data))
            ))


"""* Step 7. 모델의 검증 loop 를 정의하는 함수 생성"""

def test(data_loader):
  # 수강생 작성 코드
  # 1. 모델 객체를 테스트에 적합한 상태로 변경하세요


  # 수강생 작성 코드
  # 1. 모델의 성능을 지표를 계산에 필요한 정보 수집을 위한 변수 선언
  #   - loss 를 수집하기 위한 'test_loss' 변수 선언 및 0 으로 초기화
  #   - accuracy 계산을 위한 모델의 예측 class 와 정답 class 가 일치하는 개수를 수집하기 위한
  #     'correct' 변수 선언 및 0 으로 초기화


  # 수강생 작성 코드
  # 1. 검증데이터 에서 미니 배치를 추출하여 검증 loop를 수행하는 for문을 작성 하세요
  for ____, ____ in ____:
    # 수강생 작성 코드
    # 1. 모델에 input data를 전달하여 순전파를 수행하세요


    # 수강생 작성 코드
    # 1. loss 함수에 모델의 예측값 과 정답 정보를 전달하여 모델의 성능을 검증하기 위한 loss 값을 계산 하세요


    # logits 정보를 이용하여 모델의 예측 값과 정답 을 비교하여 모델이 맞춘 데이터의 개수를 계산
    correct += (output.argmax(dim=1) == target).type(torch.float).sum().item()

  # 'test_loss' 변수 에 누적된 배치 별 loss 의 합을 이용해 전체 loss를 계산 하세요
  test_loss /= len(test_loader)

  # 'correct' 에 누적된 배치 별 정답 개수의 합을 배치의 개수로 나누에 전체 데이터 셋의 평균 accuracy를 계산 하세요
  accuracy = 100. * correct / len(validation_loader.dataset)

  # 모델의 성능 지표를 계산 및 출력 하여 학습 진행을 체크하는 코드
  print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
        test_loss,
        accuracy))

"""* Step 8. 훈련 함수와 검증 함수를 이용한 모델의 훈련 및 검증"""
# 수강생 작성 코드
# 1. 반복문을 활용해 epoch 수 만큼 훈련 및 검증을 수행 하세요.


# 수강생 작성 코드
# 1. 테스트 데이터 셋을 이용해 모델의 성능을 확인 하세요
