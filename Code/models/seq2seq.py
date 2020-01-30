import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense


class GenericSeq2Seq:
    def __init__(self,
                 encoder_units,
                 batch_size,
                 num_timesteps,
                 vocab_size,
                 input_seq_length,
                 target_seq_length=None,
                 decoder_units=None,
                 return_state=False,
                 return_sequences=False,
                 loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy']
                 ):
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size
        self.input_seq_length = input_seq_length
        self.target_seq_length = target_seq_length
        self.return_state = return_state
        self.return_sequences = return_sequences
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def model(self):
        if self.decoder_units is None:
            self.decoder_units = self.encoder_units

        if self.target_seq_length is None:
            self.target_seq_length = self.input_seq_length

        encoder_input = Input(shape=(self.input_seq_length, self.vocab_size), batch_size=self.batch_size, name='Encoder_Input')
        encoder_lstm = LSTM(units=self.encoder_units, return_state=self.return_state, name='Encoder_LSTM')
        encoder_lstm_output = encoder_lstm(encoder_input)
        encoder_states = encoder_lstm_output[1:]

        decoder_input = Input(shape=(self.target_seq_length, self.vocab_size), batch_size=self.batch_size, name='Decoder_Input')
        decoder_lstm = LSTM(units=self.decoder_units, return_sequences=self.return_sequences, name='Decoder_LSTM')
        decoder_lstm_output = decoder_lstm(decoder_input, initial_state=encoder_states)
        decoder_dense = Dense(self.vocab_size, activation='softmax', name='Decoder_Dense')
        decoder_output = decoder_dense(decoder_lstm_output)

        model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        return model
