from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

from models.generic import GenericModel
from utils.common import reverse_dict
from utils.prediction import sample_from_softmax

import numpy as np


class GenericSeq2Seq(GenericModel):
    def __init__(self, encoder_units=None, batch_size=None, vocab_size=None, input_seq_length=None, name=None,
                 target_seq_length=None, decoder_units=None, loss='categorical_crossentropy', optimizer='adam',
                 metrics=['accuracy'], int_encoder=None):
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
        self.int_encoder = int_encoder
        if int_encoder is not None and isinstance(int_encoder, dict):
            self.int_decoder = reverse_dict(int_encoder)
        else:
            self.int_decoder = None

        self.model = None
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

        decoder_input = Input(shape=(None, self.vocab_size), batch_size=self.batch_size, name='Decoder_Input')
        decoder_lstm = LSTM(units=self.decoder_units, return_sequences=True, return_state=True, name='Decoder_LSTM')
        decoder_lstm_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
        decoder_dense = Dense(self.vocab_size, activation='softmax', name='Decoder_Dense')
        decoder_output = decoder_dense(decoder_lstm_output)

        model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.model = model

        # Encoder model
        encoder_model = Model(encoder_input, encoder_states)
        encoder_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.encoder_model = encoder_model

        # Decoder (inference) model
        decoder_state_input_h = Input(shape=(self.decoder_units,))
        decoder_state_input_c = Input(shape=(self.decoder_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_input, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_input] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        decoder_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.decoder_model = decoder_model

    def decode_sequence(self, input_seq, return_cell_states=False):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)
        # cell_states = np.zeros((1, self.decoder_units, self.input_seq_length))
        cell_states = []

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.vocab_size))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.int_encoder['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            cell_states.append(c)

            # Sample a token
            # todo: What does the -1 do here?
            # todo: Change np.argmax to sample_from_softmax
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.int_decoder[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            # todo: Remove this hardcoded max length
            if (sampled_char == '\n' or
                    len(decoded_sentence) > 100):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.vocab_size))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        if return_cell_states:
            return decoded_sentence, np.squeeze(np.array(cell_states))
        else:
            return decoded_sentence


class StateSeq2Seq(Seq2Seq):
    def build_model(self):
        if self.decoder_units is None:
            self.decoder_units = self.encoder_units

        if self.target_seq_length is None:
            self.target_seq_length = self.input_seq_length

        # Training model
        encoder_input = Input(shape=(None, self.vocab_size), batch_size=self.batch_size, name='Encoder_Input')
        encoder_lstm = LSTM(units=self.encoder_units, return_state=True, name='Encoder_LSTM')
        _, encoder_h_state, encoder_c_state = encoder_lstm(encoder_input)
        encoder_states = [encoder_h_state, encoder_c_state]

        decoder_input = Input(shape=(None, self.vocab_size), batch_size=self.batch_size, name='Decoder_Input')
        decoder_lstm = LSTM(units=self.decoder_units, return_state=True, return_sequences=True, name='Decoder_LSTM')
        decoder_lstm_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
        decoder_dense = Dense(self.vocab_size, activation='softmax', name='Decoder_Dense')
        decoder_output = decoder_dense(decoder_lstm_output)

        model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output, *encoder_states])
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.model = model

        # Encoder model
        encoder_model = Model(encoder_input, encoder_states)
        encoder_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.encoder_model = encoder_model

        # Decoder (inference) model
        decoder_state_input_h = Input(shape=(self.decoder_units,))
        decoder_state_input_c = Input(shape=(self.decoder_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_input, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_input] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        decoder_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        self.decoder_model = decoder_model
