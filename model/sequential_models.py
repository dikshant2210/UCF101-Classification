from keras.layers import LSTM, Dense
from keras.models import Sequential


def lstm_model(input_shape=(40, 2048), recurrent_units=128, num_classes=101, stack=1):
    model = Sequential()
    for _ in range(stack):
        model.add(LSTM(recurrent_units, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(recurrent_units))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model
