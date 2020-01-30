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


def char_to_int_map(max_value=9, min_value=0):
    char_to_int = {str(n): n for n in range(min_value, max_value+1)}
    n_terms = max_value - min_value + 1
    char_to_int[' '] = n_terms
    char_to_int['+'] = n_terms + 1
    char_to_int['\n'] = n_terms + 2
    return char_to_int


def input_seq_length(n_terms, n_digits):
    # Given an addition sequence with n_terms terms each with n_digits, return how many characters the (non-padded)
    # resulting input string can be (maximum possible length)
    # n_digits for each term, and n_terms - 1 "plus signs", along with an end-of-string character \n
    return n_terms * n_digits + n_terms - 1 + 1


def target_seq_length(n_terms, n_digits):
    # Given an addition sequence with n_terms terms each with n_digits, return how many characters the (non-padded)
    # resulting output string can be (maximum possible length)
    # We _don't_ add one at the end for the end-of-string character \n because we always either chop off the last digit
    # (decoder input) or chop off the first digit (decoder output)
    return n_digits + 1 + int(np.floor(np.log10(n_terms)))