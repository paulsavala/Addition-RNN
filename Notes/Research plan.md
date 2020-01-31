# Research plan
The point of this notebook is to give myself a roadmap for where to go. I will also include notes here on what I am currently working on. However, as these ideas get more developed they should get their own notes in this folder.

## Current progress
*Jan 31, 2020* - I trained two models, both with similar setups:
- 3 digits and 3 terms
- 200 epochs
- 128 batch size
- 128 hidden units (single LSTM)
- 50k samples, 10% witheld for validation, completely witheld test set
With this setup, one model was trained on the reversed data and one with the normal data. The reversed model achieved 78.3% test accuracy (I don't have the numbers from the non-reversed one saved, but it was worse). Both models were checkpointed. The next goal is to take these two models and start comparing them. What are they getting right, what are they getting wrong, where do they overlap and where are they different? Can we visualize layers using the ideas from [this paper](https://arxiv.org/pdf/1506.02078.pdf) or from [this tool](https://github.com/HendrikStrobelt/Seq2Seq-Vis)?

*Jan 30, 2020* - At this point I have a working seq2seq models which takes in two 3 digit positive integers and adds them (as strings). With 200 epochs and 128 units in the LSTM (along with reversing the input) this achieves 97% test accuracy.

## Goals
- Scale this to more digits and make sure my results still agree with [here](https://keras.io/examples/addition_rnn/)
- ~~Refactor code into PyCharm so I can use imports and reuse code~~
- Read paper ["What neural networks can reason about"](https://arxiv.org/pdf/1905.13211.pdf) to train and gain a theoretical understanding of the situation
- Start learning about LSTM variants (attention, transformer, etc.)
- Visualize the model using either [this paper](https://arxiv.org/pdf/1506.02078.pdf) or [this tool](https://github.com/HendrikStrobelt/Seq2Seq-Vis)
