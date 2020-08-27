import numpy as np


# todo: Rewrite these first three functions. I'm confusing the ideas of X and y with input and
#  target which makes it all hard to follow
def example_prediction(model, X, y):
    # Given a compiled model and input and target data, generate a prediction (used for testing)
    X_singleton = X[0].reshape(1, *X[0].shape)
    y_singleton = y[0].reshape(1, *y[0].shape)

    output = model.predict([X_singleton, y_singleton], batch_size=1)
    return output


def sample_from_softmax(output):
    # Given the softmax output from a model, sample from it (as opposed to np.argmax)
    squeezed_output = np.squeeze(output)
    sampled_output = np.random.choice(np.arange(len(squeezed_output)), p=squeezed_output)
    return sampled_output


def sample_from_lstm_softmax(output):
    # Given the softmax output from an LSTM with shape (n_samples, n_timesteps, vocab_size),
    # return a vector of size (n_samples, n_timesteps) which has the vocab sampled
    decoded_output = np.zeros(shape=(output.shape[0], output.shape[1]))
    for i, sample in enumerate(output):
        for t, ts in enumerate(sample):
            value = sample_from_softmax(ts)
            decoded_output[i, t] = value
    return decoded_output


def evaluate_one(model, X, y):
    # Given a compiled model and input and target data, compare the predictions with the correct values
    model_output = example_prediction(model, X, y)
    sampled_pred = sample_from_softmax(model_output)
    # todo: Make this output nicer to get a better idea what's actually being predicted/used
    print(f'X = {X}')
    print(f'y = {y}')
    print(f'Pred = {sampled_pred}')


def pprint_metrics(metrics, metrics_names, return_pprint=False):
    pretty_metrics = '\n'.join(f'{n} = {v:.4f}' for n, v in list(zip(metrics_names, metrics)))
    print(pretty_metrics)
    if return_pprint:
        return pretty_metrics