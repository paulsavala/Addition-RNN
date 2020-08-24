from models.seq2seq import Seq2Seq
from utils.integers import char_to_int_map, input_seq_length, target_seq_length
from utils.common import reverse_dict
from utils.training import format_targets
from utils.prediction import pprint_metrics
from data_gen.integer_addition import generate_samples, generate_all_samples

from tensorflow.keras.optimizers import Adam


class Config:
    n_terms = 2
    n_digits = 2
    test_size = 10**2
    reverse = False
    encoder_units = 16


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
    X_test, y_test = generate_all_samples(n_terms=Config.n_terms,
                                          n_digits=Config.n_digits,
                                          int_encoder=Mappings.char_to_int,
                                          one_hot=True,
                                          reverse=Config.reverse)

    # Evaluate it on the test set (sanity check)
    print('Test set metrics:')
    input_test_target, output_test_target = format_targets(y_test)
    preds = model.model.predict(x=[X_test, input_test_target], verbose=0)
    pass
