# Research plan
The point of this notebook is to give myself a roadmap for where to go. I will also include notes here on what I am currently working on. However, as these ideas get more developed they should get their own notes in this folder.

## Current progress
*Aug 27, 2020* - Back at it! The visualization discussed last time didn't give me any great insight into what's going on. So instead I focused on looking at the predictions the model makes correctly and incorrectly. In particular, I trained models everywhere from 2^0 to 2^64 hidden units in the encoder, and compared their predictions. I see the following general principles:
- Models seem to learn "on the diagonal." By which I mean, they learn all of the pairs which sum to (say) 115. You can see this in the plots in the [examining predictions notebook](../Notebooks/Examining%20predictions.ipynb), especially in the "binary" heatmaps, which simply classify each sum as being correct or incorrect. From the non-binary heatmaps (which use the predicted sum minus the actual sum as their "heat"), we see that they learn in "waves". On one diagonal they consistently overpredict (perhaps because they're acting like a step function?), then slowly move towards correct prediction, then finally to underpredicting, before repeating this process over and over.
- In general, models learn most quickly and most effectively sums which sum to a number "in the middle". So for instance, in the 2 terms 2 digits model (so summing from 0+0 to 99+99), the range of possible sums is 0 to 198. The models perform best with terms that sum to roughly the midpoint. This holds true even when you account for how many sums actually sum to that value (for instance, only one sum sums to 0, but many sum to 100). This could potentially just be a factor of seeing more training examples.
- Interestingly, with sums at the midpoint or past (so from 99 to 198) they follow a very predictable pattern, where the model progressively and smoothly gets worse and worse at learning them. However, for sums *before* the midpoint, it's all over the place. There is no discrenable pattern.

*Feb 10, 2020* - I got visualization working. The setup is that the decoder model has cell states, and I run through prediction of an input sequence one timestep at a time. At each timestep I record the cell states for each hidden unit in the LSTM. I then plot these on top of the numbers in the input sequence they're looking at to see what each cell "thinks" of that digit. It's interesting, but not super useful right now. The problems are:
- I need at least 32 units to get decent (95%+) results, which makes it harder to see what each unit is doing
- It's just not obvious what (if any) pattern there is
I need to keep trying other things, and combine these visualizations with other techniques and tools to get a better grasp of what's going on. Here are some things I can try:
- Try training the smaller models (2dig_2term for instance) with less hidden units. If I can get good enough accuracy then this could be easier to work with
- Is it possible to selectively deactivate certain units in a cell? If so I could deactivate them one at a time to get an idea of the importance of each particular cell. That way I can focus on what the most important cells are learning.
- Some cells seem to show very little variation in their responses. Maybe programatically remove these from the visualizations just to make it easier to see what's going on
- Compare cell activity across multiple samples. That way I can see which behave similarly regardless of the sample (and thus are doing global things like counting the number of digits seen), and which vary (and thus are doing local things like looking for zeros, nines, etc.)

*Jan 31, 2020* - I trained two models, both with similar setups:
- 3 digits and 3 terms
- 200 epochs
- 128 batch size
- 128 hidden units (single LSTM)
- 50k samples, 10% witheld for validation, completely witheld test set
With this setup, one model was trained on the reversed data and one with the normal data. The reversed model achieved 78.3% test accuracy (I don't have the numbers from the non-reversed one saved, but it was worse). Both models were checkpointed. The next goal is to take these two models and start comparing them. What are they getting right, what are they getting wrong, where do they overlap and where are they different? Can we visualize layers using the ideas from [this paper](https://arxiv.org/pdf/1506.02078.pdf) or from [this tool](https://github.com/HendrikStrobelt/Seq2Seq-Vis)?

*Jan 30, 2020* - At this point I have a working seq2seq models which takes in two 3 digit positive integers and adds them (as strings). With 200 epochs and 128 units in the LSTM (along with reversing the input) this achieves 97% test accuracy.

## Goals
- ~~Scale this to more digits and make sure my results still agree with [here](https://keras.io/examples/addition_rnn/)~~ (Done Feb 2020)
- ~~Refactor code into PyCharm so I can use imports and reuse code~~ (Done 1/30/2020)
- Read paper ["What neural networks can reason about"](https://arxiv.org/pdf/1905.13211.pdf) to train and gain a theoretical understanding of the situation
- Start learning about LSTM variants (attention, transformer, etc.)
- ~~Visualize the model using either [this paper](https://arxiv.org/pdf/1506.02078.pdf) or [this tool](https://github.com/HendrikStrobelt/Seq2Seq-Vis)~~ (Done Feb 2020)

## Todo
- Rewrite my models in KMM so that they inherit directly from a Keras Model. That way I don't have to keep writing my_model.model.keras_model_fcn
- ~~Setup KMM to save model params as well. So inputs to the `__init__` call so that later I can load those params separately to see what I set~~ (Done Feb 2020)
