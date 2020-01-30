import numpy as np


def one_hot(n, max_value):
    # One-hots a positive integer n where n <= max_value
    one_hot_n = np.zeros(max_value)
    one_hot_n[n] = 1
    return one_hot_n


def undo_one_hot(v):
    # If an integer is one-hot encoded using the one_hot function above, return the integer n
    return np.argmax(v)


def one_hot_matrix(M, max_value):
    # Given a matrix M of size (n_samples, n_ints) return the matrix one-hotted. The return matrix is
    # of size (n_samples, n_ints, max_value)
    n_samples, seq_length = M.shape
    M_oh = np.array([one_hot(r, max_value) for r in np.array(M).flatten()]).reshape(
        (n_samples, seq_length, max_value))

    # In case this is a target vector, we don't want to include an unnecessary axis
    return np.squeeze(M_oh)


def _generate_sample(n_terms, n_digits, int_encoder=None, reverse=False):
    # Generate a sample of the form "number_1+number_2+...+number_{n_terms}=answer".
    # Each number_i has n_digits digits
    # If a dictionary is passed for int_encoder then use the it to convert characters to integers (so for instance
    # convert '3' to 3 or '+' to 12)
    x = []
    for _ in range(n_terms):
        x.append(np.random.randint(10 ** n_digits))

    y = np.sum(x)

    x_str = '+'.join(str(n) for n in x)
    y_str = str(y)

    # Pad x so that is always has the same length. It should be of length digit_length for each digit,
    # plus num_terms - 1 "plus" signs
    x_str = x_str.rjust(n_terms * n_digits + n_terms - 1)
    max_target_digits = n_digits + 1 + int(np.floor(np.log10(n_digits)))
    y_str = y_str.rjust(max_target_digits)

    if reverse:
        x_str = x_str[::-1]

    x_str += '\n'
    y_str += '\n'

    x_list = list(x_str)
    y_list = list(y_str)

    if int_encoder is not None:
        assert isinstance(int_encoder, dict), 'int_encoder must be a dictionary mapping characters to integers'
        x_list = [int_encoder[c] for c in x_list]
        y_list = [int_encoder[c] for c in y_list]

    return x_list, y_list


def generate_samples(n_samples, n_terms=2, n_digits=2, int_encoder=None, one_hot=False, reverse=False):
    # Generate n_samples examples of addition problems as defined in _generate_samples above
    X = []
    y = []
    for _ in range(n_samples):
        x_sample, y_sample = _generate_sample(n_terms, n_digits, int_encoder, reverse)
        X.append(x_sample)
        y.append(y_sample)

    X = np.array(X)
    y = np.array(y)

    if one_hot:
        X = one_hot_matrix(X, len(int_encoder))
        y = one_hot_matrix(y, len(int_encoder))

    return X, y