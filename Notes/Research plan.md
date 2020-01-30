# Research plan
The point of this notebook is to give myself a roadmap for where to go. I will also include notes here on what I am currently working on. However, as these ideas get more developed they should get their own notes in this folder.

## Current progress
Jan 30, 2020 - At this point I have a working seq2seq models which takes in two 3 digit positive integers and adds them (as strings). With 200 epochs and 128 units in the LSTM (along with reversing the input) this achieves 97% test accuracy.

## Goals
- Scale this to more digits and make sure my results still agree with [here](https://keras.io/examples/addition_rnn/)
- Refactor code into PyCharm so I can use imports and reuse code
- See what results are achieved in [here](https://arxiv.org/abs/1904.01557.pdf) and duplicate them
- Read paper ["What neural networks can reason about"](https://arxiv.org/pdf/1905.13211.pdf) to train and gain a theoretical understanding of the situation
- Start researching how to visualize/understand what LSTMs are learning
