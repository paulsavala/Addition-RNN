# General LSTM Notes
The purpose of this note is to cover some topics around the practical implementation of LSTMs in Keras that I could not easily
find the answers to initially. These notes work together with the notebook ["Learning LSTMs"](notebooks/Learning_LSTMs.ipynb). 

## What LSTMs return
According to the [Keras RNN guide](https://www.tensorflow.org/guide/keras/rnn) "by default, the output of a RNN layer contains
a single output vector per sample. This vector is the RNN cell output corresponding to the last timestep." As discussed in
[Colah's excellent writeup](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), the hidden state of an LSTM encodes
information about the input vector, along with all the previous hidden states. The size of this final hidden state is simply
the number of units in the LSTM (as a side note, I will pretty much always ignore batch sizes. In reality, the size is
`(batch_size, units)`, but this is true essentialy all of the time, so I'll just omit it). 

Having said that, LSTMs also offer two parameters to alter the returned values. One is `return_state` and the other is
`return_sequences`. By default both are set to `False`.

### Return_state
`return_state` specifies that the LSTM should also return the final cell state (again, see Colah's notes). However, somewhat
confusingly, this causes the LSTM to now return three values. This is confusing because what is the third value? We know
about the final hidden state and the final cell state, so what else is left? The answer is that the final hidden state is
actually returned twice. It is returned as the LSTM's output, and then again as the second returned object. The following
code block illustrates this:

```
lstm_input = Input(shape=(num_timesteps, num_features))
lstm = LSTM(units=num_units, return_state=return_state, return_sequences=return_sequences)
lstm_output = lstm(lstm_input)

assert type(lstm_output) == list
assert len(lstm_output) == 3

lstm_output, final_hidden_state, final_cell_state = lstm_output

assert lstm_output == final_hidden_state
```

It may seem strange that the LSTM would return its final hidden state twice. However, the reason for this is that 
`return_sequences` alters the output of the LSTM. Therefore, if someone still wanted access to the final hidden state, they
could specify `return_state=True` and still have access to it.

### Return_sequences
So what does `return_sequences=True` do? It specifies that the LSTM output (so the first object returned from the LSTM)
should be a vector containing *all* hidden states. Therefore it is a vector of size `(timesteps, units)`.
