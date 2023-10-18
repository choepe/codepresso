import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers, models

tf.__version__

auto_mpg_dataset_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin', 'Car Name']
raw_dataset = pd.read_csv(auto_mpg_dataset_url, delim_whitespace=True, names=column_names)
raw_dataset.dtypes

dataset = raw_dataset.drop(labels='Car Name', axis=1)
dataset.dtypes
dataset = dataset[(dataset['Horsepower'] != '?')]
dataset['Horsepower'] = dataset['Horsepower'].astype(np.float64)

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

len(train_dataset)
len(test_dataset)

train_stats = train_dataset.describe()
train_stats.pop('MPG')
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def norm(x):
    normed_data = (x - train_stats['mean']) / train_stats['std']
    return normed_data

train_dataset = norm(train_dataset)
test_dataset = norm(test_dataset)

train_dataset.describe()

model = models.Sequential()
model.add(layers.Dense(units=64, activation='relu', input_shape=(len(train_dataset.keys()),)))
model.add(layers.Dense(units=32, activation='relu'))
model.add(layers.Dense(units=1))
model.summary()

model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
history = model.fit(
    x=train_dataset, y=train_labels,
    batch_size=16,
    epochs=100,
    validation_split=0.2
)

hist = pd.DataFrame(history.history)
loss = history.history['loss']
val_loss = history.history['val_loss']

mae = history.history['mae']
val_mae = history.history['val_mae']

epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.ylim([0, 20])
plt.legend()
plt.show()

plt.plot(epochs, mae, label='Training mae')
plt.plot(epochs, val_mae, label='Validation mae')
plt.title('Training and validation mae')
plt.xlim([1, 100])
plt.ylim([0, 20])
plt.legend()
plt.show()

test_loss, test_mae = model.evaluate(x=test_dataset, y=test_labels)
test_predictions = model.predict(test_dataset).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, 50])
plt.ylim([0, 50])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()