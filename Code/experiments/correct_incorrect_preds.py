from models.seq2seq import Seq2Seq
from utils.integers import char_to_int_map, input_seq_length, target_seq_length, undo_one_hot_matrix
from utils.common import reverse_dict
from utils.file_io import list_to_csv
from utils.training import format_targets
from utils.prediction import pprint_metrics
from data_gen.integer_addition import generate_all_samples, generate_samples

import re


class Config:
    n_terms = 2
    n_digits = 2
    train_size = 5 * 10 ** 3
    test_size = 10 ** 3
    validation_split = 0
    epochs = 200
    reverse = False
    encoder_units = 32
    batch_size = 64
    allow_less_terms = False
    uniform_samples = True
    generalize_to_n_terms = False


class Mappings:
    char_to_int = char_to_int_map()
    int_to_char = reverse_dict(char_to_int)


if __name__ == '__main__':
    # Load the pretrained model
    model_name = 'basic_addition'
    model_name += f'_{Config.n_terms}term_{Config.n_digits}dig_{Config.encoder_units}units'
    if getattr(Config, 'allow_less_terms', False):
        model_name += '_varterms'
    if getattr(Config, 'generalize_to_n_terms', False):
        model_name += f'_generalize{Config.generalize_to_n_terms}'
    if getattr(Config, 'uniform_samples', False):
        model_name += '_uniform_samples'
    if getattr(Config, 'reverse', False):
        model_name += '_reversed'

    target_model = Seq2Seq(name=model_name)
    target_model.load_model(version=1, load_attributes=True)

    n_terms = max(getattr(Config, 'generalize_to_n_terms', 0), Config.n_terms)

    # Hack to work around build_model destroying the loaded weights
    model = Seq2Seq(name='basic_addition_states',
                    encoder_units=Config.encoder_units,
                    batch_size=1,
                    input_seq_length=input_seq_length(n_terms, Config.n_digits),
                    target_seq_length=target_seq_length(n_terms, Config.n_digits),
                    vocab_size=len(Mappings.char_to_int),
                    int_encoder=Mappings.char_to_int
                    )
    model.build_model()
    model.load_weights(target_model, load_attributes=True)

    if Config.generalize_to_n_terms:
        # Generate some test data to visualize
        X_test, y_test = generate_samples(n_samples=Config.test_size,
                                          n_terms=Config.generalize_to_n_terms,
                                          n_digits=Config.n_digits,
                                          int_encoder=Mappings.char_to_int,
                                          one_hot=True,
                                          reverse=Config.reverse,
                                          allow_less_terms=Config.allow_less_terms)
    else:
        X_test, y_test = generate_all_samples(n_terms=Config.n_terms,
                                              n_digits=Config.n_digits,
                                              int_encoder=Mappings.char_to_int,
                                              one_hot=True,
                                              reverse=Config.reverse)

    # Evaluate it on the test set and decode the predictions
    print('Test set metrics:')
    input_test_target, output_test_target = format_targets(y_test)
    test_metrics = model.model.evaluate(x=[X_test, input_test_target], y=output_test_target, verbose=0)
    pprint_metrics(test_metrics, model.model.metrics_names)
    print('\n\n')

    decoded_X = undo_one_hot_matrix(X_test, Mappings.int_to_char)
    decoded_y = undo_one_hot_matrix(y_test, Mappings.int_to_char)

    preds_fname = f'basic_addition_{Config.n_terms}term_{Config.n_digits}dig_{Config.encoder_units}units'
    if Config.generalize_to_n_terms:
        preds_fname += f'_generalize{Config.generalize_to_n_terms}'
    if Config.uniform_samples:
        preds_fname += '_uniform'
    correct_buffer = []
    incorrect_buffer = []
    for i in range(X_test.shape[0]):
        if i > 0 and i % 100 == 0:
            print(f'{i}/{X_test.shape[0]}')
            print(f'{100*len(correct_buffer)/i:.1f}% correct')
            print('-'*20)
            list_to_csv(correct_buffer, preds_fname + '_correct.csv', headers='X_true, y_true')
            list_to_csv(incorrect_buffer, preds_fname + '_incorrect.csv', headers='X_true, y_pred, y_true')
            correct_buffer = []
            incorrect_buffer = []
        X_true = decoded_X[i]
        y_true = decoded_y[i]
        pred = model.decode_sequence(X_test[i])
        y_true_val = int(re.search(r'\d+', y_true).group(0))
        pred_val = int(re.search(r'\d+', pred).group(0))
        X_pattern = r'\+'.join([r'\d+' for _ in range(n_terms)])
        # todo: Hack to support variable number of terms, fix this by proper striping/replacing
        if re.search(X_pattern, X_true) is not None:
            X_true_cleaned = re.search(X_pattern, X_true).group(0)
        else:
            X_true_cleaned = re.search(r'\d+\+\d+', X_true).group(0)
        if y_true_val == pred_val:
            correct_buffer.append(f'{X_true_cleaned}, {y_true_val}')
        else:
            incorrect_buffer.append(f'{X_true_cleaned}, {pred_val}, {y_true_val}')

