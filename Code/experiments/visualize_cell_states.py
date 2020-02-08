import numpy as np
import csv
from matplotlib import pyplot as plt

from utils.common import sigmoid


if __name__ == '__main__':
    input_seqs = []
    with open('experiments/cell_states/2term_2dig/input.csv') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            input_seqs.append(row[-1])

    cell_states = np.load('experiments/cell_states/2term_2dig/cell_states.npy', allow_pickle=True)
    print(f'Shape: {cell_states.shape}')

    single_sample = input_seqs[0]
    single_cell_state = cell_states[0]

    assert len(single_sample) == single_cell_state.shape[0], "Cell timesteps and input sequence length don't match"

    # Iterate through the units
    height_step = 1 / single_cell_state.shape[1]
    width_step = 1 / single_cell_state.shape[0]
    for cell in range(single_cell_state.shape[1]):
        cell_data = single_cell_state[:, cell]
        for ts in range(single_cell_state.shape[0]):
            plt.text(x=ts * width_step, y=cell * height_step, s=single_sample[ts],
                     backgroundcolor=(1, 0, 0, sigmoid(cell_data[ts])))
    plt.show()