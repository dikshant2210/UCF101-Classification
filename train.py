import pandas as pd
import pickle as pkl
import numpy as np
from model.sequential_models import lstm_model


batch_size = 32
epochs = 5

train = pd.read_csv('data/train.csv', names=('path', 'category'))
validation = pd.read_csv('data/validation.csv', names=('path', 'category'))
test = pd.read_csv('data/test.csv', names=('path', 'category'))


target_dict = dict()
for index, category in enumerate(train['category'].unique()):
    target_dict[category] = index

print('Training on {} samples.'.format(train.shape[0]))
print('Validating on {} samples.'.format(validation.shape[0]))
print('Testing on {} samples.'.format(test.shape[0]))


def train_generator():
    while True:
        for start in range(0, train.shape[0], batch_size):
            x_batch = list()
            y_batch = list()
            end = min(start + batch_size, train.shape[0])
            train_path_batch = train['path'][start: end]
            train_category_batch = train['category'][start: end]
            for path, category in zip(train_path_batch.values, train_category_batch.values):
                with open(path, 'rb') as file:
                    array = pkl.load(file)
                    x_batch.append(array)
                target = np.zeros(shape=(101,))
                target[target_dict[category]] = 1
                y_batch.append(target)
            x_batch = np.array(x_batch, np.float64)
            y_batch = np.array(y_batch, np.float64)
            yield x_batch, y_batch


def valid_generator():
    while True:
        for start in range(0, validation.shape[0], batch_size):
            x_batch = list()
            y_batch = list()
            end = min(start + batch_size, validation.shape[0])
            valid_path_batch = validation['path'][start: end]
            valid_category_batch = validation['category'][start: end]
            for path, category in zip(valid_path_batch.values, valid_category_batch.values):
                with open(path, 'rb') as file:
                    array = pkl.load(file)
                    x_batch.append(array)
                target = np.zeros(shape=(101,))
                target[target_dict[category]] = 1
                y_batch.append(target)
            x_batch = np.array(x_batch, np.float64)
            y_batch = np.array(y_batch, np.float64)
            yield x_batch, y_batch


model = lstm_model()
print(model.summary())

model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(train)) / float(batch_size)),
                    epochs=epochs,
                    verbose=1,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(validation)) / float(batch_size)))

model.save_weights('weights/model.h5')
