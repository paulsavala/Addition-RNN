# Research plan
The point of this notebook is to give myself a roadmap for where to go. I will also include notes here on what I am currently working on. However, as these ideas get more developed they should get their own notes in this folder.

## Current progress
*Nov 24, 2020* - I implemented curriculum learning by having it go through one-digit sums (sums of form `0+0+x`, `0+x+0` and `x+0+0`), then two-digit sums, then three-digit sums. I also implemented the ability to reverse the output (`y`) when predicting, so that it predicts the ones place first, then the tens, etc. I am still running analysis with these and will post results once they're done. From my initial analysis it looks like it drastically improved the situation with the attention mechanism not doing anything useful. I also fixed training with a reversed `y` (it was stopping as soon as it predicted a newline character, which was the first thing it was _supposed to_ predict!). It's training now.

Going forward, here are some notes/tasks:
1. I'm starting to think again about the publication side of this. I need to have concrete results to demonstrate the effectiveness. That means I need to outline some concrete goals for myself, and then write code to evaluate any model on train against these goals. That way I can directly see if I'm closer to accomplishing those or not (which I'm not currently doing). Some goals could include:
 - Overall accuracy
 - Avg difference between predicted and correct (as integers)
 - Results on repeated sums (`1+1+1`, `2+2+2`, etc.)
 - Results on commuted sums
 - Results on subsums. For example, if it correctly computes `10+20+0` and `0+20+5`, can it also compute `10+20+5`?

*Nov 2, 2020* - I've tried a number of things since last time. The major one is that I changed the encoder-decode model to use attention, and also to use a bidirectional GRU on the encoder side. I did bidirectional since addition is commutative. When looking at the results, we can see some basic attention mechanism usage going on for 2 term 2 digits sums. It's not perfect, but you can sort of tease out what's going on. However, when looking at 3 terms 2 digits, that's all gone. The attention mechanism basically just looks at the "end of sequence" character "\n". Why is this happening? Here are some thoughts:
1. It's using the hidden states in the encoder instead of attention. Since there are a small number of examples, it's just able to memorize. We can also see that the accuracy drops greatly for 3 terms 2 digits, so that may also reflect what's going on.
2. It's likely not learning subsums at all. My guess is that this is the case because training is random. I'm just randomly picking samples to train on. So sometimes it trains early on three digit sums before it has seen the corresponding 2 digit sums. I should try curriculum learning (training in a structured manner) to address this.
3. Right now I'm having predictions made from left to right (e.g. hundreds place, then tens place, etc.). What about doing it in reverse? Predicting the ones place, then tens place, etc? Does this change the results?
My next steps should be training on the other stratified training sets and seeing how that does. Then, I can follow that up with curriculum learning. The goal is still to see both an attention mechanism that "behaves properly", and hidden states that are (hopefully) sensible.

*Sep 12, 2020* - I wanted to list out some thoughts I had today about directions for work on this project, along with some observations now that I've run a number of experiments and begun collecting and analyzing data.
- This "learning along the diagonal" is really how we expect NN to learn. After all, if you had a cats vs dogs image classifier, you would expect that it would take the class of cats and learn the features that relate to being a cat. Learning along the diagonal is exactly this. It's taking a sum, and (presumably) looking for the features that cause series to sum to that value.
- In the paper that inspired this work (the Google paper on NN reasoning about math) they hypothesized that maybe these are learning subsums. Perhaps one way to test this is to train a network on sums of the form "x+y+0" where x and y vary as usual. Then, it could be evaluated on inputs of the form "x+y" (with appropriate padding). If they truly are learning subsums, then they should learn "x+y" (presumably the networks understand that "+0" does nothing, but I should look into this).
- My guess is that it's really _not_ learning subsums. It's just learning (memorizing?) features that make a series sum to a certain value. One way to test this is to fix the training set (save it) and see how well the model generalizes to series in the held out test set, but with the same sum as others in the training set. In general, how many times does the model need to see series summing to a given value to then generalize that knowledge? 
- One problem(?) with making the output of these networks be the value the series sums to is that there just aren't that many training examples for each sum. For example, when summing to (say) 100, there's only 10 series with 2 terms and 2 digits that sum to it. So how much can it really learn general features? It might be interesting to train separate models to predict each digit. So one model predicts the one's place, one predicts the ten's place, etc. The problem(?) with this is that it sort of violates the spirit of this kind of work. Ideally we should be able to feed a plain text math problem to a network and have it output the right answer, without having to build in any assumptions on the form of the output (i.e. that it's a digit with some number of places). 
 
*Sep 10, 2020* - Today I wanted to address my point in my Aug 27 entry that "In general, models learn most quickly and most effectively sums which sum to a number 'in the middle'". I hypothesized that this may just be due to there being more series which sum to these values. So I wrote a new data generation function (`_generate_uniform_samples` in `data_gen/integer_addition.py`) which generates samples uniformly w.r.t. the sum. I only did it for the 32 unit network. The results still show learning along the diagonal, and also still show a preference for learning sums "in the middle". This _could_ still be due to seeing more _diverse_ series which sum to these values. For example, now there are (roughly) an equal number of series summing to 2 and there are to 100. However, there's really only three series summing to 2 (0+2, 1+1, 2+0), it's just that the network sees these over and over. On the other hand, the fact that the network only needs to "learn" such a small number of examples makes you think it should do so easily, which is _not_ the case. Looking forward, here are some more things I can still try/work on:
- Run these same experiments (with uniform sampling) for other numbers of encoder units and see if the results stay consistent.
- Compare results from non-uniform sampling to uniform sampling for the same models.
- Is it possible to "see" the "learning along the diagonal" in the cell activation visualizations I made earlier? Maybe find two series summing to the same value and see if there is any commonality that is not present in a different series. (I just tried this with 74+18 and 72+20, vs some other sum, and indeed there is a lot of similarity between 74+18 and 72+20, and a lot of that similarity dissapears when looking at some other random series.)
- Since it seems to learn on the diagonal, is it actually learning, or just memorizing? For instance, if it is trained on 10+90, 11+89, 12+88, ..., 18+82, can it also figure out that 19+91=100? Other other sums on the same diagonal, without explicitly seeing them during training?

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

## Other papers
- [Survey of automatic math word problem solving](https://arxiv.org/abs/1808.07290)
- [Application of BERT to math word problems](https://arxiv.org/pdf/1909.00109.pdf)
- [Overview of curriculum learning approaches](https://arxiv.org/pdf/1904.03626.pdf)
- [Extension of Transformer model to solve math problems](https://arxiv.org/pdf/1910.06611.pdf)
