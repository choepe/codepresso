import torch
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

# Step 1. put Input tensor and Target tensor
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
type(train_images), type(train_labels)
train_images.shape, train_labels.shape
train_labels[0]

# Step 2. 2D Tensor to 1D Tensor
train_images = train_images.reshape((60000, 28*28))
test_images = test_images.reshape((10000, 28*28))

'''
# Step 3. Create a batch pipeline
-학습데이터를 torch.Tensor 객체로 변환
-변환 된 torch.Tensor 객체를 DataSet 객체에 연결
-Batch 구성을 위한 DataLoader 객체 생성
'''
train_x = torch.Tensor(train_images)
train_y = torch.LongTensor(train_labels)
test_x = torch.Tensor(test_images)
test_y = torch.LongTensor(test_labels)

train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
#x, y = next(iter(train_dataset))

batch_size = 64
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
#x, y = next(iter(train_dataloader))

'''
# Step 4. DNN(MLP) model
-torch.nn.Sequential를 이용하여 비어있는 모델 객체 생성
-Sequential 모델 객체의 .append() 메서드를 이용하여 layer 추가
-summary API를 이용하여 정의한 모델 확인
'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.nn.Sequential()
model.append(module=torch.nn.Linear(in_features=784, out_features=256, bias=True))
model.append(module=torch.nn.ReLU())
model.append(module=torch.nn.Linear(in_features=256, out_features=10, bias=True))
model.to(device)
summary(model=model, input_size=(784,), batch_size=batch_size)

'''
# Step 5. 학습 관련 객체 생성
-torch.nn 모듈에 구현된 API를 이용해 분류를 위한 손실함수 객체를 생성
-torch.optim 모듈에 구현된 API로 옵티마이저 객체를 생성
'''
#for param in model.parameters():
#    print(param.shape)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(params=model.parameters(), lr=1e-3)

# Step 6. 모델의 학습 loop를 정의하는 함수 생성
history = {
    'loss': [],
    'val_loss': [],
    'acc': [],
    'val_acc': []
}
x, y = next(iter(train_dataloader))
x = x.to(device)
y = y.to(device)
model.train()
logits = model(x)