import numpy as np
from utils.integers import one_hot_matrix, input_seq_length, target_seq_length
from itertools import product


def _generate_sample(n_terms, n_digits):
    # Generate a sample of the form "number_1+number_2+...+number_{n_terms}=answer"
    x = []
    for _ in range(n_terms):
        x.append(np.random.randint(10 ** n_digits - 1))

    y = np.sum(x)

    x_str = '+'.join(str(n) for n in x)
    y_str = str(y)
    return x_str.strip(), y_str.strip()


def _format_sample(x_str, y_str, n_terms, n_digits, int_encoder=None, reverse=False):
    # Format a sample of the form "number_1+number_2+...+number_{n_terms}=answer".
    # Each number_i has n_digits digits
    # If a dictionary is passed for int_encoder then use the it to convert characters to integers (so for instance
    # convert '3' to 3 or '+' to 12)

    # Pad x so that is always has the same length. We subtract one because we don't yet account for the \n character
    max_input_digits = input_seq_length(n_terms, n_digits) - 1
    x_str = x_str.rjust(max_input_digits)
    max_target_digits = target_seq_length(n_terms, n_digits)
    y_str = y_str.rjust(max_target_digits)

    if reverse:
        x_str = x_str[::-1]

    # Prepend an end-of-sequence character \n and for the target append a start-of-sequence character \t
    x_str += '\n'
    y_str = '\t' + y_str + '\n'

    x_list = list(x_str)
    y_list = list(y_str)

    if int_encoder is not None:
        assert isinstance(int_encoder, dict), 'int_encoder must be a dictionary mapping characters to integers'
        x_list = [int_encoder[c] for c in x_list]
        y_list = [int_encoder[c] for c in y_list]

    return x_list, y_list


def generate_samples(n_samples, n_terms=2, n_digits=2, int_encoder=None, one_hot=False, reverse=False):
    # Generate n_samples examples of addition problems as defined in _generate_sample above
    X = []
    y = []
    for _ in range(n_samples):
        x_str, y_str = _generate_sample(n_terms, n_digits)
        x_sample, y_sample = _format_sample(x_str, y_str, n_terms, n_digits, int_encoder, reverse)
        X.append(x_sample)
        y.append(y_sample)

    X = np.array(X)
    y = np.array(y)

    if one_hot:
        X = one_hot_matrix(X, len(int_encoder))
        y = one_hot_matrix(y, len(int_encoder))

    return X, y


def generate_all_samples(n_terms=2, n_digits=2, int_encoder=None, one_hot=False, reverse=False):
    # Generate all possible integer addition problems
    X = []
    y = []

    x_all = range(10 ** n_digits)
    x_cartesian = list(product(x_all, repeat=n_terms))
    for x in x_cartesian:
        x_str = '+'.join([str(a) for a in x])
        y_str = str(sum(x))
        x_str = x_str.strip()
        y_str = y_str.strip()
        x_sample, y_sample = _format_sample(x_str, y_str, n_terms, n_digits, int_encoder, reverse)
        X.append(x_sample)
        y.append(y_sample)

    assert len(X) == 10 ** (n_digits * n_terms), "Looks like you didn't generate all possible problems..."

    X = np.array(X)
    y = np.array(y)

    if one_hot:
        X = one_hot_matrix(X, len(int_encoder))
        y = one_hot_matrix(y, len(int_encoder))

    return X, y