# Sequence-to-Sequence
The point of these notes is to address questions/things I learned from implementing a sequence-to-sequence (seq2seq) model.

## Overview
The general process is as follows:
- Split the model into an encoder and a decoder. 
- Encoder:
    - The encoder takes the input sequence as its input and outputs its internal state (both cell and hidden states, but only at the final timestep)
- Decoder:
    - The decoder uses as initial states the output states from the encoder
    - It uses as input the *target* sequence and is trained to predict the next timestep in the target sequence (again)
    
Note then that the entire purpose of the encoder is to generate a vector space representation of the input in terms of the 
hidden and cell states of the encoder. That is, the encoder serves to compress the information in the input sequencer.

The role of the decoder is to use this compression as information to decode a target sequence. Note that we *do not* use the input sequence in the decoder. This is because all information from the target sequence is (in theory) already held in the hidden states from the encoder. Thus the role of the decoder is to use that information to work directly with the target sequence. 

## Return_sequences
In order for the model to work correctly it must be the case that the decoder LSTM has `return_sequences=True`. Why? It seemed initially to me that the decoder LSTM output is just whatever it is, and as long as it's passed along to the final dense layer then the dense layer can just do its thing. But running it without that setup (in training mode) will result in an error. 

I don't have a 100% confident answer, but I think I have an idea. Recall that by default the LSTM only returns the final hidden state. The final hidden state is the hidden state corresponding to the last timestep. That is, it's the hidden state corresponding to *only the last character*. Now this is a bit confusing, because in fact each hidden state is (in part) determined by all of the previous hidden states, and thus the previous timesteps, and so all previous characters. But regardless, each hidden state `h_t` only directly relate to the timestep `t`, and thus the input character `c_t`. I don't feel like this fully answers my question, but it's a start. I think too that this was previously addressed using either `TimeDistributed` or `RepeatVector` as mentioned [here](https://machinelearningmastery.com/learn-add-numbers-seq2seq-recurrent-neural-networks/). But Keras has since updated the Dense layer to learn how to hande 3d outputs. More reading needs to be done here.

## Training
In my notebook "Learning seq2seq" (and to a lesser extend "Text generation") I go through several slightly different training setups to try and understand how they affect the performance of the model. In particular I try the following things:
- Embedding layer vs one-hot ("Text generation" notebook)
- 1 unit in LSTM vs 128
- 200 epochs vs 30 epochs
- Reversing the input

Let's discuss the results of each.

### Embedding layer vs one-hot
One way to handle integer-encoded data is to one-hot it. That way we end up with vectors of the shape `(timesteps, seq_length, vocab_size)` as desired. However, as done [here](https://www.tensorflow.org/tutorials/text/text_generation) they use an Embedding layer as an alternative to one-hotting. I was curious to see the difference. In terms of code, the only real model change is to add an embedding layer and pick the embedding dimension (somewhat arbitrarily). The one advantage of an embedding layer from a code perspective is that you don't have to deal with one-hotting (and reversing one-hotting). Not a huge deal, but kind of nice.

I trained two models, one with and one without an embedding layer, both for 5 epochs with a batch size of 64, a single LSTM unit, and a 40% validation set. I achieved 44.89% validation accuracy *without* an embedding layer, and 47.03% *with* an embedding layer. Same exact data for both. Anecdotally, the model *with* an embedding layer took approximately 10s extra per epoch to train (~230s vs ~220s). Strangely enough my validation loss (and accuracy) actually *dropped* each epoch when I had an embedding layer. So perhaps it needed more epochs to get into its groove. It's interesting though that it still outperformed one-hotting.

### (1 unit and 30 epochs) vs (128 units and 200 epochs) 
Now let's turn to the seq2seq setup. I trained on 10k samples with 30% used as the validation set. Using just one unit in the LSTM resulted in a validation accuracy of 43.26%. Unfortunately I made some other tweaks besides just switching to 128 units. In particular I dropped it to 5k samples, completely set aside the validation set, and trained for 200 epochs. After 30 epochs I had a *training* accuracy of 49.72%. After all 200 epochs I evaluated the accuracy on my withheld validation set and had a validation accuracy of 65.07%. It's hard to say how much of that came from just doing more eopchs, and how much came from the units. Although it also seems obvious that more units would greatly help accuracy.

### Reversing the input
At this point I had largely duplicated the setup in [this Keras script](https://keras.io/examples/addition_rnn/). I had changed the data, batch size, epochs and hidden units to all match what they had. However, they claimed 99% test accuracy, and I was only getting 65%. The only thing I hadn't done was reverse the input as they suggested. As a reminder, this means reversing *just* the input string and *not* the target. So for instance, rather than using `x='123+456\n', y='579\n'` we would use `rev_x='\n654+321', y='579\n'`. This seems very strange to me, but they claim it is useful. Indeed, running with exactly the same setup as I achieved 65% accuracy but with instead reversing the input the model jumped to 97.37% test accuracy. Incredible! This is definitely worth exploring. I haven't seen a good explanation of why this phenomenon occurs. the best I've found is in [this paper](https://arxiv.org/pdf/1409.3215.pdf) (the most famous seq2seq paper) which says the following:
>  We found it extremely valuable to reverse the order of the words of the input sentence. So for example, instead of mapping the sentence a, b, c to the sentence α, β, γ, the LSTM is asked to map c, b, a to α, β, γ, where α, β, γ is the translation of a, b, c. This way, a is in close proximity to α, b is fairly close to β, and so on, a fact that makes it easy for SGD to “establish communication” between the input and the output.
> While we do not have a complete explanation to this phenomenon, we believe that it is caused by the introduction of many short term dependencies to the dataset. Normally, when we concatenate a source sentence with a target sentence, each word in the source sentence is far from its corresponding word in the target sentence. As a result, the problem has a large “minimal time lag” [link](https://papers.nips.cc/paper/1215-lstm-can-solve-hard-long-time-lag-problems.pdf). By reversing the words in the source sentence, the average distance between corresponding words in the source and target language is unchanged. However, the first few words in the source language are now very close to the first few words in the target language, so the problem’s minimal time lag is greatly reduced. Thus, backpropagation has an easier time “establishing communication” between the source sentence and the target sentence, which in turn results in substantially improved overall performance.

> Initially, we believed that reversing the input sentences would only lead to more confident predictions in the early parts of the target sentence and to less confident predictions in the later parts. However, LSTMs trained on reversed source sentences did much better on long sentences than LSTMs trained on the raw source sentences (see sec. 3.7), which suggests that reversing the input sentences results in LSTMs with better memory utilization.

These explanation seems plausible, but I'd like to have a more concrete grasp on what's going on.
