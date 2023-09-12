import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

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
