from pathlib import Path
import numpy as np


class Data:
    base_dir = Path('data')
    random = base_dir / Path('random')
    sum_strat = base_dir / Path('sum_strat')
    uniform_sum = base_dir / Path('uniform_sum')

    data_groups = [random, sum_strat, uniform_sum]


    @classmethod
    def load(cls, data, n_terms, n_digits):
        assert data in cls.data_groups, 'data must be a Data class variable'
        term_dig_dir = Path(f'{n_terms}term_{n_digits}digs')

        dir = data / term_dig_dir
        X_train = np.load(dir / Path('X_train.npy'))
        X_test = np.load(dir / Path('X_test.npy'))
        y_train = np.load(dir / Path('y_train.npy'))
        y_test = np.load(dir / Path('y_test.npy'))

        return X_train, X_test, y_train, y_test
