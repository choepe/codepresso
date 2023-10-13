import torch
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
train_x.to(device)
train_y.to(device)
test_x.to(device)
test_y.to(device)


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
logits.shape
batch_loss = loss(logits, y)

model.parameters()
for name, param in model.named_parameters():
    print(name)
    print(param.shape)
print(model.get_parameter('2.bias'))
print(model.get_parameter('2.bias').grad)
batch_loss.backward()
print(model.get_parameter('2.bias').grad)
optimizer.step()
print(model.get_parameter('2.bias'))
optimizer.zero_grad()

def train(epoch):
    model.train()

    train_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.to(device)  # 데이터를 GPU로 이동
        target = target.to(device)  # 타겟을 GPU로 이동

        output = model(data)
        batch_loss = loss(output, target)

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        correct += (output.argmax(dim=1) == target).type(torch.float).sum().item()
        train_loss += batch_loss.item()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(epoch, (batch_idx + 1) * len(data), len(train_dataloader.dataset), 100. * (batch_idx + 1) / len(train_dataloader), batch_loss.item(), 100. * correct / ((batch_idx + 1) * len(data))))
    history['loss'].append(train_loss / len(train_dataloader))
    history['acc'].append(100. * correct / len(train_dataloader.dataset))

# Step 7. 모델의 검증 loop를 정의하는 함수 생성
def test(data_loader):
    model.eval()

    test_loss = 0
    correct = 0

    for data, target in data_loader:
        data = data.to(device)  # 데이터를 GPU로 이동
        target = target.to(device)  # 타겟을 GPU로 이동

        output = model(data)
        test_loss += loss(output, target).item()
        correct += (output.argmax(dim=1) == target).type(torch.float).sum().item()

    test_loss /= len(test_dataloader)
    accuracy = 100. * correct / len(data_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(test_loss, accuracy))

    history['val_loss'].append(test_loss)
    history['val_acc'].append(accuracy)

# Step 8. 훈련 함수와 검증 함수를 이용한 모델의 훈련 및 검증
epochs = 30

for epoch in range(1, epochs + 1):  # 변경: range(1, 30) -> range(1, epochs + 1)
    train(epoch)
    test(test_dataloader)

epochs = range(len(history['loss']))

plt.plot(epochs, history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history['val_loss'], 'b', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

plt.plot(epochs, history['acc'], 'bo', label='Training accuracy')
plt.plot(epochs, history['val_acc'], 'b', label='validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

# Step 9. 테스트 데이터 셋을 이용한 성능 검증
test(test_dataloader)
test_image = test_images[0]
prediction = model(torch.Tensor(test_image).to(device))
prediction.argmax()
plt.figure()
plt.imshow(test_images[0].reshape(28, 28))
plt.colorbar()
plt.grid(False)
plt.show()