from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

from models.generic import GenericModel

import numpy as np


class GenericSeq2Seq(GenericModel):
    def __init__(self, encoder_units=None, batch_size=None, vocab_size=None, input_seq_length=None, name=None,
                 target_seq_length=None, decoder_units=None, loss='categorical_crossentropy', optimizer='adam',
                 metrics=['accuracy']):
        super().__init__(name)
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.input_seq_length = input_seq_length
        self.target_seq_length = target_seq_length
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        self.encoder_model = None
        self.decoder_model = None


class Seq2Seq(GenericSeq2Seq):
    def build_model(self):
        if self.decoder_units is None:
            self.decoder_units = self.encoder_units

        if self.target_seq_length is None:
            self.target_seq_length = self.input_seq_length

        encoder_input = Input(shape=(self.input_seq_length, self.vocab_size), batch_size=self.batch_size, name='Encoder_Input')
        encoder_lstm = LSTM(units=self.encoder_units, return_state=True, name='Encoder_LSTM')
        encoder_lstm_output = encoder_lstm(encoder_input)
        encoder_states = encoder_lstm_output[1:]

        decoder_input = Input(shape=(self.target_seq_length, self.vocab_size), batch_size=self.batch_size, name='Decoder_Input')
        decoder_lstm = LSTM(units=self.decoder_units, return_sequences=True, name='Decoder_LSTM')
        decoder_lstm_output = decoder_lstm(decoder_input, initial_state=encoder_states)
        decoder_dense = Dense(self.vocab_size, activation='softmax', name='Decoder_Dense')
        decoder_output = decoder_dense(decoder_lstm_output)

        model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        self.model = model


class StateSeq2Seq(GenericSeq2Seq):
    def build_model(self):
        if self.decoder_units is None:
            self.decoder_units = self.encoder_units

        if self.target_seq_length is None:
            self.target_seq_length = self.input_seq_length

        encoder_input = Input(shape=(self.input_seq_length, self.vocab_size), batch_size=self.batch_size, name='Encoder_Input')
        encoder_lstm = LSTM(units=self.encoder_units, return_state=True, name='Encoder_LSTM')
        _, encoder_h_state, encoder_c_state = encoder_lstm(encoder_input)
        encoder_states = [encoder_h_state, encoder_c_state]

        decoder_input = Input(shape=(self.target_seq_length, self.vocab_size), batch_size=self.batch_size, name='Decoder_Input')
        decoder_lstm = LSTM(units=self.decoder_units, return_sequences=True, name='Decoder_LSTM')
        decoder_lstm_output = decoder_lstm(decoder_input, initial_state=encoder_states)
        decoder_dense = Dense(self.vocab_size, activation='softmax', name='Decoder_Dense')
        decoder_output = decoder_dense(decoder_lstm_output)

        model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output, *encoder_states])
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        self.model = model

    def load_weights(self, target_model=None, weights_file=None):
        assert target_model is not None or weights_file is not None, 'You must specify either a target model to load weights from or a weights file'
        assert target_model is None or weights_file is None, 'Specify target_model or weights_file, not both'
        assert self.model is not None, 'First run build_model()'

        if target_model is not None:
            self.model.set_weights(target_model.model.get_weights())
        else:
            self.model.load_weights(weights_file)
