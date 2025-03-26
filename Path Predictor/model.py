import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def create_model(sequence_length, num_features):
    input_shape = (sequence_length, num_features)  # batch size, length of sequence, number of features

    # Construct RNN model with LSTM layer for memory and Dense layer for pos_x, pos_z output
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=input_shape, recurrent_dropout=0.5, return_sequences=False))
    model.add(Dense(2)) # Number of outputs: which is predicted pos_x and pos_z
    model.compile(optimizer='adam', loss='mse')

    return model

def train_model(model, x_train, y_train, epochs, batch_size):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

