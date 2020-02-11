from models.seq2seq import Seq2Seq
from utils.integers import char_to_int_map, input_seq_length, target_seq_length, undo_one_hot_matrix
from utils.common import reverse_dict
from utils.training import format_targets
from utils.prediction import pprint_metrics
from data_gen.integer_addition import generate_samples

import numpy as np
from pathlib import Path


class Config:
    n_terms = 4
    n_digits = 2
    test_size = 10**2
    reverse = False
    batch_size = 128
    encoder_units = 32


class Mappings:
    char_to_int = char_to_int_map()
    int_to_char = reverse_dict(char_to_int)


if __name__ == '__main__':
    # Load the pretrained model
    model_name = 'basic_addition'
    model_name += f'_{Config.n_terms}term_{Config.n_digits}dig'
    if Config.reverse:
        model_name += '_reversed'

    target_model = Seq2Seq(name=model_name)
    target_model.load_model(version=1, load_attributes=True)

    # Hack to work around build_model destroying the loaded weights
    model = Seq2Seq(name='basic_addition_states',
                         encoder_units=Config.encoder_units,
                         batch_size=1,
                         input_seq_length=input_seq_length(Config.n_terms, Config.n_digits),
                         target_seq_length=target_seq_length(Config.n_terms, Config.n_digits),
                         vocab_size=len(Mappings.char_to_int),
                         int_encoder=Mappings.char_to_int
                         )
    model.build_model()
    model.load_weights(target_model, load_attributes=True)

    # Generate some test data to visualize
    X_test, y_test = generate_samples(n_samples=Config.test_size,
                                      n_terms=Config.n_terms,
                                      n_digits=Config.n_digits,
                                      int_encoder=Mappings.char_to_int,
                                      one_hot=True,
                                      reverse=Config.reverse)

    # Evaluate it on the test set (sanity check)
    print('Test set metrics:')
    input_test_target, output_test_target = format_targets(y_test)
    test_metrics = model.model.evaluate(x=[X_test, input_test_target], y=output_test_target, verbose=0)
    pprint_metrics(test_metrics, model.model.metrics_names)
    print('\n\n')

    # Decode an input repeatedly
    num_samples = 100
    input_samples = []
    decoded_samples = []
    for i in range(num_samples):
        X_singleton = X_test[i].reshape(1, *X_test[i].shape)
        y_singleton = y_test[i].reshape(1, *y_test[i].shape)
        X_pred, cell_states = model.decode_sequence(X_singleton, get_input_cell_states=True)
        input_samples.append(undo_one_hot_matrix(X_singleton, Mappings.int_to_char)[0])
        decoded_samples.append(cell_states)

    decoded_samples = np.array(decoded_samples)
    print(f'Saving with shape {decoded_samples.shape}')
    cell_states_dir = f'experiments/cell_states/{Config.n_terms}term_{Config.n_digits}dig'
    if Config.reverse:
        cell_states_dir += '_reversed'
    cell_states_dir = Path(cell_states_dir)
    if not cell_states_dir.exists():
        cell_states_dir.mkdir()
    np.save(cell_states_dir / Path('cell_states.npy'), decoded_samples)
    with open(cell_states_dir / Path('input.csv'), 'w') as f:
        for i in range(num_samples):
            f.write(input_samples[i])

    # We don't need this model anymore, it was just used to save cell states
    model.delete_model()