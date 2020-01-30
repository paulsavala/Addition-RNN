from models.seq2seq import SimpleSeq2Seq
from utils.integers import char_to_int_map, input_seq_length, target_seq_length
from utils.common import reverse_dict
from utils.training import format_targets
from utils.prediction import pprint_metrics
from data_gen.integer_addition import generate_samples

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

import os


class Config:
    n_terms = 2
    n_digits = range(2, 4)
    train_size = 5 * 10**2
    validation_split = 0.1
    test_size = 10**2
    epochs = 3
    encoder_units = 1
    batch_size = 1


class Mappings:
    char_to_int = char_to_int_map()
    int_to_char = reverse_dict(char_to_int)


train_data_X = dict()
train_data_y = dict()
train_data_X_rev = dict()
train_data_y_rev = dict()

test_data_X = dict()
test_data_y = dict()
test_data_X_rev = dict()
test_data_y_rev = dict()
for d in Config.n_digits:
    X_train, y_train = generate_samples(n_samples=Config.train_size,
                                        n_digits=d,
                                        int_encoder=Mappings.char_to_int,
                                        one_hot=True)
    X_test, y_test = generate_samples(n_samples=Config.test_size,
                                      n_digits=d,
                                      int_encoder=Mappings.char_to_int,
                                      one_hot=True)
    X_train_rev, y_train_rev = generate_samples(n_samples=Config.train_size,
                                                n_digits=d,
                                                int_encoder=Mappings.char_to_int,
                                                one_hot=True,
                                                reverse=True)
    X_test_rev, y_test_rev = generate_samples(n_samples=Config.test_size,
                                              n_digits=d,
                                              int_encoder=Mappings.char_to_int,
                                              one_hot=True,
                                              reverse=True)

    train_data_X[d] = X_train
    train_data_y[d] = y_train
    test_data_X[d] = X_test
    test_data_y[d] = y_test

    train_data_X_rev[d] = X_train_rev
    train_data_y_rev[d] = y_train_rev
    test_data_X_rev[d] = X_test_rev
    test_data_y_rev[d] = y_test_rev


if __name__ == '__main__':
    # Run the experiment once for each digit length, as well as being reversed or not
    for d in Config.n_digits:
        for reverse in [True, False]:
            # Define the model
            model_class = SimpleSeq2Seq(encoder_units=Config.encoder_units,
                                        batch_size=Config.batch_size,
                                        input_seq_length=input_seq_length(Config.n_terms, d),
                                        target_seq_length=target_seq_length(Config.n_terms, d),
                                        vocab_size=len(Mappings.char_to_int)
                                        )
            model = model_class.model()

            # Create a callback that saves the model's weights
            checkpoint_path = f"experiments/increase_digits/logs/cp_{d}_rev_{reverse}.ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)

            cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          save_weights_only=True,
                                          verbose=1)
            csv_logger = CSVLogger(f'experiments/logs/{d}_rev_{reverse}.csv', append=True, separator=';')

            # Begin training
            print(f'Experiment: {d} digits with reverse = {reverse}')
            if reverse:
                X_train = train_data_X_rev[d]
                y_train = train_data_y_rev[d]
                X_test = test_data_X_rev[d]
                y_test = test_data_y_rev[d]
            else:
                X_train = train_data_X[d]
                y_train = train_data_y[d]
                X_test = test_data_X[d]
                y_test = test_data_y[d]

            # Format the input and output
            input_train_target, output_train_target = format_targets(y_train)
            model.fit([X_train, input_train_target],
                      output_train_target,
                      epochs=Config.epochs,
                      callbacks=[cp_callback, csv_logger])

            # Evaluate on the test set
            input_test_target, output_test_target = format_targets(y_test)
            test_metrics = model.evaluate([X_test, input_test_target],
                                          output_test_target,
                                          verbose=0)
            pprint_metrics(test_metrics, model.metrics_names)
            print(f'{"="*20}\n')