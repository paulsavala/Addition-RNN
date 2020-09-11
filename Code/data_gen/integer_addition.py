import numpy as np
from utils.integers import one_hot_matrix, input_seq_length, target_seq_length
from itertools import product
import random


def _generate_sample(n_terms, n_digits, allow_less_terms=False):
    # Generate a sample of the form "number_1+number_2+...+number_{n_terms}=answer"
    x = []
    if allow_less_terms:
        for _ in range(np.random.randint(2, n_terms + 1)):
            x.append(np.random.randint(10 ** n_digits - 1))
    else:
        for _ in range(n_terms):
            x.append(np.random.randint(10 ** n_digits - 1))

    y = np.sum(x)

    x_str = '+'.join(str(n) for n in x)
    y_str = str(y)
    return x_str.strip(), y_str.strip()


def _generate_sample_from_y(n_terms, n_digits, y):
    # Generates a sample which sums to y (used to uniformly distribute the sums)
    x = []
    while len(x) < n_terms - 1:
        # Don't allow it to pick a number causing sum(x) to exceed y, but also subject
        # to the restriction of n_digits.

        # Also, don't allow it to pick such a small number that it would be impossible
        # for the remaining terms to be chosen to sum to y (for example, if y = 150 and
        # n_terms = 2, n_digits = 2, we can't pick 49, or else you would need 101 to sum
        # to y.
        y_upper_bound = y - np.sum(x)
        n_digits_upper_bound = 10 ** n_digits - 1
        upper_bound = min([y_upper_bound, n_digits_upper_bound])
        lower_bound = (y - np.sum(x) - (10 ** n_digits - 1) * (n_terms - len(x) - 1))
        lower_bound = max([0, lower_bound])

        if upper_bound > 0:
            x.append(np.random.randint(lower_bound, upper_bound + 1))
        else:
            x.append(0)
    x.append(y - np.sum(x))
    random.shuffle(x)

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


def generate_samples(n_samples, n_terms=2, n_digits=2, int_encoder=None, one_hot=False, reverse=False,
                     allow_less_terms=False, uniform=False):
    if uniform:
        X, y = _generate_uniform_samples(n_samples, n_terms, n_digits, int_encoder, one_hot, reverse)
    else:
        X, y = _generate_samples(n_samples, n_terms, n_digits, int_encoder, one_hot, reverse, allow_less_terms)
    return X, y

# Generate n_samples examples of addition problems as defined in _generate_sample above


def _generate_samples(n_samples, n_terms=2, n_digits=2, int_encoder=None, one_hot=False, reverse=False, allow_less_terms=False):
    # Generate n_samples examples of addition problems as defined in _generate_sample above
    X = []
    y = []
    for _ in range(n_samples):
        x_str, y_str = _generate_sample(n_terms, n_digits, allow_less_terms=allow_less_terms)
        x_sample, y_sample = _format_sample(x_str, y_str, n_terms, n_digits, int_encoder, reverse)
        X.append(x_sample)
        y.append(y_sample)

    X = np.array(X)
    y = np.array(y)

    if one_hot:
        X = one_hot_matrix(X, len(int_encoder))
        y = one_hot_matrix(y, len(int_encoder))

    return X, y


def _generate_uniform_samples(n_samples, n_terms=2, n_digits=2, int_encoder=None, one_hot=False, reverse=False):
    max_sum = (10**n_digits - 1) * n_terms
    possible_sums = range(max_sum + 1)

    X = []
    y = []
    for _ in range(n_samples):
        x_str, y_str = _generate_sample_from_y(n_terms, n_digits, np.random.choice(possible_sums))
        x_sample, y_sample = _format_sample(x_str, y_str, n_terms, n_digits, int_encoder, reverse)
        assert len(x_sample) == 6, f'x_str = {x_str}, x_sample = {x_sample}'
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