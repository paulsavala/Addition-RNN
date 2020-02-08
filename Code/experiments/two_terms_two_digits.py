from models.seq2seq import Seq2Seq
from utils.integers import char_to_int_map, input_seq_length, target_seq_length
from utils.common import reverse_dict
from utils.training import format_targets
from utils.prediction import pprint_metrics
from data_gen.integer_addition import generate_samples

from tensorflow.keras.optimizers import Adam


class Config:
    n_terms = 2
    n_digits = 2
    train_size = 5 * 10**3
    test_size = 10**2
    epochs = 100
    reverse = False


class Mappings:
    char_to_int = char_to_int_map()
    int_to_char = reverse_dict(char_to_int)


X_train, y_train = generate_samples(n_samples=Config.train_size,
                                    n_terms=Config.n_terms,
                                    n_digits=Config.n_digits,
                                    int_encoder=Mappings.char_to_int,
                                    one_hot=True,
                                    reverse=Config.reverse)
X_test, y_test = generate_samples(n_samples=Config.test_size,
                                  n_terms=Config.n_terms,
                                  n_digits=Config.n_digits,
                                  int_encoder=Mappings.char_to_int,
                                  one_hot=True,
                                  reverse=Config.reverse)


if __name__ == '__main__':
    # Define the model
    model_name = 'basic_addition_2term_2dig'
    if Config.reverse:
        model_name += '_reversed'

    model = Seq2Seq(name=model_name,
                    encoder_units=128,
                    batch_size=128,
                    input_seq_length=input_seq_length(Config.n_terms, Config.n_digits),
                    target_seq_length=target_seq_length(Config.n_terms, Config.n_digits),
                    vocab_size=len(Mappings.char_to_int),
                    int_encoder=Mappings.char_to_int
                    )
    model.build_model()

    # Format the input and output
    input_train_target, output_train_target = format_targets(y_train)
    model.train(X=[X_train, input_train_target],
                y=output_train_target,
                optimizer=Adam,
                loss='categorical_crossentropy',
                metrics=['accuracy'],
                epochs=Config.epochs)

    # Evaluate on the test set
    input_test_target, output_test_target = format_targets(y_test)
    test_metrics = model.model.evaluate(x=[X_test, input_test_target], y=output_test_target, verbose=0)
    pprint_metrics(test_metrics, model.model.metrics_names)

    model_notes = 'Two terms with two digits each. Trained over 100 epochs with 5*10^3 samples'
    if Config.reverse:
        model_notes += ' REVERSED.'

    model.save_model(notes=model_notes)
