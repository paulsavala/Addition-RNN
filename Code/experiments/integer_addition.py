from models.seq2seq import Seq2Seq
from utils.integers import char_to_int_map, input_seq_length, target_seq_length
from utils.common import reverse_dict
from utils.data import Data
from utils.training import format_targets
from utils.prediction import pprint_metrics
from data_gen.integer_addition import generate_samples

from tensorflow.keras.optimizers import Adam


class Config:
    n_terms = 2
    n_digits = 2
    train_size = 5 * 10**3
    test_size = 10**3
    validation_split = 0
    epochs = 100
    reverse = False
    encoder_units = 32
    batch_size = 64
    allow_less_terms = False
    uniform_samples = False
    generalize_to_n_terms = False
    saved_data = Data.uniform_sum


class Mappings:
    char_to_int = char_to_int_map()
    int_to_char = reverse_dict(char_to_int)


if getattr(Config, 'saved_data'):
    # Load the saved data
    X_train, X_test, y_train, y_test = Data.load(Config.saved_data, Config.n_terms, Config.n_digits)
else:
    X_train, y_train = generate_samples(n_samples=Config.train_size,
                                        n_terms=Config.n_terms,
                                        n_digits=Config.n_digits,
                                        int_encoder=Mappings.char_to_int,
                                        one_hot=True,
                                        reverse=Config.reverse,
                                        allow_less_terms=Config.allow_less_terms,
                                        uniform=Config.uniform_samples)
    X_test, y_test = generate_samples(n_samples=Config.test_size,
                                      n_terms=Config.n_terms,
                                      n_digits=Config.n_digits,
                                      int_encoder=Mappings.char_to_int,
                                      one_hot=True,
                                      reverse=Config.reverse,
                                      allow_less_terms=Config.allow_less_terms,
                                      uniform=Config.uniform_samples)

if getattr(Config, 'generalize_to_n_terms', 0) > Config.n_terms:
    X_generalized_test, y_generalized_test = generate_samples(n_samples=Config.test_size,
                                                              n_terms=Config.generalize_to_n_terms,
                                                              n_digits=Config.n_digits,
                                                              int_encoder=Mappings.char_to_int,
                                                              one_hot=True,
                                                              reverse=Config.reverse,
                                                              allow_less_terms=Config.allow_less_terms)


if __name__ == '__main__':
    # Define the model
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
    if getattr(Config, 'saved_data', False):
        model_name += f'_{Config.saved_data.parts[-1]}'

    n_terms = max(getattr(Config, 'generalize_to_n_terms', 0), Config.n_terms)

    model = Seq2Seq(name=model_name,
                    encoder_units=Config.encoder_units,
                    batch_size=Config.batch_size,
                    input_seq_length=input_seq_length(n_terms, Config.n_digits),
                    target_seq_length=target_seq_length(n_terms, Config.n_digits),
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
                epochs=Config.epochs,
                validation_split=Config.validation_split)

    # Evaluate on the test set
    print('Test set metrics:')
    input_test_target, output_test_target = format_targets(y_test)
    test_metrics = model.model.evaluate(x=[X_test, input_test_target], y=output_test_target, verbose=0)
    pretty_metrics = pprint_metrics(test_metrics, model.model.metrics_names, return_pprint=True)

    if Config.generalize_to_n_terms:
        # Evaluate on the test set with more terms, if applicable
        print(f'Test set metrics (generalized to {Config.generalize_to_n_terms} terms):')
        input_test_target, output_test_target = format_targets(y_generalized_test)
        test_metrics = model.model.evaluate(x=[X_generalized_test, input_test_target], y=output_test_target, verbose=0)
        pretty_metrics = pprint_metrics(test_metrics, model.model.metrics_names, return_pprint=True)

    model_notes = f'''
        {Config.n_terms} terms with {Config.n_digits} digits each.\n
        Epochs: {Config.epochs}\n
        Train size: {Config.train_size}\n
        Test size: {Config.test_size}\n
        Validation split: {Config.validation_split}\n
        Encoder units: {Config.encoder_units}\n
        Batch size: {Config.batch_size}\n
        Reversed: {Config.reverse}\n
        Allow less terms: {Config.allow_less_terms}
        '''
    model_notes += f'\n\nTest metrics:\n{pretty_metrics}'

    model.save_model(notes=model_notes)
