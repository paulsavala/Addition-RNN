from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input


class GenericLSTM:
    def __init__(self,
                 units,
                 batch_size,
                 num_timesteps,
                 vocab_size,
                 return_state=False,
                 return_sequences=False,
                 loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy']):
        self.units = units
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def model(self):
        raise NotImplementedError


class SimpleLSTM(GenericLSTM):
    # A simple LSTM used primarily for testing
    def model(self):
        model = Sequential()
        model.add(LSTM(units=self.units, input_shape=(self.num_timesteps, self.vocab_size)))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        return model


class SequentialLSTM(GenericLSTM):
    # A simple LSTM, but defined using the Keras Functional API
    def model(self):
        lstm_input = Input(shape=(self.num_timesteps, self.vocab_size))
        lstm = LSTM(units=self.units, return_state=self.return_state, return_sequences=self.return_sequences)
        lstm_output = lstm(lstm_input)
        model = Model(lstm_input, lstm_output)

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        return model