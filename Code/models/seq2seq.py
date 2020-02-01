from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

from models.generic import GenericModel

import numpy as np


class GenericSeq2Seq:
    def __init__(self,
                 encoder_units=None,
                 batch_size=None,
                 vocab_size=None,
                 input_seq_length=None,
                 target_seq_length=None,
                 decoder_units=None,
                 loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy']
                 ):
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


class KMMGenericSeq2Seq(GenericModel):
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


class KMMSeq2Seq(KMMGenericSeq2Seq):
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


class SimpleSeq2Seq(GenericSeq2Seq):
    def model(self):
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
        return model

    def restore_model(self, loaded_model):
        # Recreate the model after loading it from a previously saved one (model.save(...)) using
        # tf.keras.models.load_model
        encoder_inputs = loaded_model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = loaded_model.layers[2].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = loaded_model.input[1]  # input_2
        decoder_state_input_h = Input(shape=(self.decoder_units,), name='input_3')
        decoder_state_input_c = Input(shape=(self.decoder_units,), name='input_4')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = loaded_model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = loaded_model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.vocab_size))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence


class Seq2SeqCombinedModels(GenericSeq2Seq):
    def model(self):
        if self.decoder_units is None:
            self.decoder_units = self.encoder_units

        if self.target_seq_length is None:
            self.target_seq_length = self.input_seq_length

        encoder_input = Input(shape=(self.input_seq_length, self.vocab_size), batch_size=self.batch_size, name='Encoder_Input')
        encoder_lstm = LSTM(units=self.encoder_units, return_state=True, name='Encoder_LSTM')
        encoder_lstm_output = encoder_lstm(encoder_input)
        encoder_states = encoder_lstm_output[1:]
        self.encoder_model = Model(inputs=encoder_input, outputs=encoder_states)

        decoder_input = Input(shape=(self.target_seq_length, self.vocab_size), batch_size=self.batch_size, name='Decoder_Input')
        decoder_lstm = LSTM(units=self.decoder_units, return_sequences=True, name='Decoder_LSTM')
        decoder_lstm_output = decoder_lstm(decoder_input, initial_state=encoder_states)
        decoder_dense = Dense(self.vocab_size, activation='softmax', name='Decoder_Dense')
        decoder_output = decoder_dense(decoder_lstm_output)
        self.decoder_model = Model(inputs=[decoder_input, encoder_states], outputs=decoder_output)

        model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        return model