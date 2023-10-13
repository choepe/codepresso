import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# preprocessing
auto_mpg_dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin', 'Car Name']

raw_dataset = pd.read_csv(auto_mpg_dataset_url, delim_whitespace=True, names=column_names)
raw_dataset.dtypes
dataset = raw_dataset.drop(labels='Car Name', axis=1)
rows_to_drop = dataset[dataset['Horsepower'] == '?'].index
dataset = dataset.drop(rows_to_drop)
dataset['Horsepower'] = dataset['Horsepower'].astype(np.float64)
train_dataset = dataset.sample(frac=0.8)
test_dataset = dataset.drop(labels=train_dataset.index)
len(train_dataset), len(test_dataset)

train_dataset.dtypes
train_stats = train_dataset.describe()
train_stats.drop(labels='MPG', axis=1, inplace=True)

train_stats = train_stats.transpose()

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

train_mean = train_stats['mean']
train_std = train_stats['std']

train_dataset = (train_dataset - train_mean) / train_std
test_dataset = (test_dataset - train_mean) / train_std

train_dataset.describe()

# pipeline
train_x = torch.Tensor(train_dataset.values.astype(np.float32))
test_x = torch.Tensor(test_dataset.values.astype(np.float32))
train_x[0]
train_y = torch.Tensor(train_labels.values).unsqueeze(dim=1)
test_y = torch.Tensor(test_labels.values).unsqueeze(dim=1)
train_x = train_x.to(device)
train_y = train_y.to(device)
test_x = test_x.to(device)
test_y = test_y.to(device)


train_x.shape, train_y.shape

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

x, y = next(iter(train_dataset))
x
y

batch_size = 8
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


x, y = next(iter(train_loader))
x.shape, y.shape

# model
model = torch.nn.Sequential(
    torch.nn.Linear(7, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1)
)
model = model.to(device)

#model = torch.nn.Sequential()
#model.append(module=torch.nn.Linear(in_features=7, out_features=64))
#model.append(module=torch.nn.ReLU())
#model.append(module=torch.nn.Linear(in_features=64, out_features=32))
#model.append(module=torch.nn.ReLU())
#model.append(module=torch.nn.Linear(in_features=32, out_features=16))
#model.append(module=torch.nn.ReLU())
#model.append(module=torch.nn.Linear(in_features=16, out_features=1))
#model = model.to(device)
summary(model=model, input_size=(7,), batch_size=batch_size)

loss = torch.nn.MSELoss()
mae = torch.nn.L1Loss()
optimizer = torch.optim.RMSprop(params=model.parameters(), lr=1e-3)

history = {
    'mse': [],
    'val_mse': [],
    'mae': [],
    'val_mae': []
}

def train(epoch):
    # 모델 객체를 학습에 적합한 상태로 변경
    model.train()

    train_mae = 0
    train_mse = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        # 순전파 수행
        output = model(x)
        # loss 값 계산
        batch_mse = loss(output, y)
        train_mse += batch_mse.item()
        # 파아미터들이 loss에 미치는 영향도 계산
        batch_mse.backward()
        # 영향도 값을 이용하여 파라미터 업데이트
        optimizer.step()
        #optimizer 초기화
        optimizer.zero_grad()
        # mae 계산
        batch_mae = mae(output, y)
        train_mae += batch_mae.item()

        if batch_idx%5 == 0:
            print('Train Epoch: {} [{}/{} ({: .0f}%)]\tMAE: {: .6f}\tMSE: {: .6f}'.format(
                epoch,
                (batch_idx+1) * len(x),
                len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader),
                batch_mae,
                batch_mse
            ))

    history['mse'].append(train_mse / len(train_loader))
    history['mae'].append(train_mae / len(train_loader))

def test(data_loader):
    # 모델 객체를 테스트에 적합한 상태로 변경
    model.eval()
    test_mae = 0
    test_mse = 0

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        # 모델의 순전파 수행
        output = model(x)

        test_mse += loss(output, y).item()
        test_mae += mae(output, y).item()

    print('\nTest set: Average MAE: {: .4f}\t MSE: {: .4f}\n'.format(
        test_mae,
        test_mse
    ))

    history['val_mse'].append(test_mse)
    history['val_mae'].append(test_mae)

epoch = 20

for epoch_idx in range(1, epoch+1):
    train(epoch_idx)
    test(test_loader)

epochs = range(len(history['mae']))

plt.plot(epochs, history['mae'], 'bo', label='Training mae')
plt.plot(epochs, history['val_mae'], 'b', label='validation mae')
plt.titel('Training and validation loss')
plt.legend()

plt.show()

epochs = range(len(history['mse']))

plt.plot(epochs, history['mse'], 'ro', label='Training mae')
plt.plot(epochs, history['val_mse'], 'r', label='validation mse')
plt.titel('Training and validation loss')
plt.legend()

plt.show()

test(test_loader)