from models.seq2seq import Seq2Seq, StateSeq2Seq
from utils.integers import char_to_int_map, input_seq_length, target_seq_length, undo_one_hot_matrix
from utils.common import reverse_dict
from utils.training import format_targets
from utils.prediction import pprint_metrics
from data_gen.integer_addition import generate_samples


class Config:
    n_terms = 2
    n_digits = 2
    test_size = 10**2
    reverse = False
    batch_size = 128


class Mappings:
    char_to_int = char_to_int_map()
    int_to_char = reverse_dict(char_to_int)


X_test, y_test = generate_samples(n_samples=Config.test_size,
                                  n_terms=Config.n_terms,
                                  n_digits=Config.n_digits,
                                  int_encoder=Mappings.char_to_int,
                                  one_hot=True,
                                  reverse=Config.reverse)


if __name__ == '__main__':
    # Load the pretrained model
    model_name = 'basic_addition_2term_2dig'
    if Config.reverse:
        model_name += '_reversed'

    target_model = Seq2Seq(name=model_name)
    target_model.load_model(version=1)

    # Evaluate it on the test set (sanity check, should be around 80% test accuracy for the "normal" model
    # and 99% for the reversed model)
    input_test_target, output_test_target = format_targets(y_test)
    test_metrics = target_model.model.evaluate(x=[X_test, input_test_target], y=output_test_target, verbose=0)
    pprint_metrics(test_metrics, target_model.model.metrics_names)

    # # Create a new model that returns states and load weights from the pretrained model
    # # todo: Once I add the ability to save and load model params, then change this to directly load them from the pretrained model
    model = StateSeq2Seq(name='basic_addition_viz',
                         encoder_units=128,
                         batch_size=1,
                         input_seq_length=input_seq_length(Config.n_terms, Config.n_digits),
                         target_seq_length=target_seq_length(Config.n_terms, Config.n_digits),
                         vocab_size=len(Mappings.char_to_int),
                         int_encoder=Mappings.char_to_int
                         )
    model.build_model()
    model.load_weights(target_model)

    # Decode an input
    X_singleton = X_test[0].reshape(1, *X_test[0].shape)
    y_singleton = y_test[0].reshape(1, *y_test[0].shape)
    X_pred, cell_states = model.decode_sequence(X_singleton, return_cell_states=True)
    print(f'Input sequence: {undo_one_hot_matrix(X_singleton, Mappings.int_to_char)}')
    print(f'Ground truth: {undo_one_hot_matrix(y_singleton, Mappings.int_to_char)}')
    print(f'Prediction: {X_pred}')
    print(f'Cell states: {cell_states.shape}')
