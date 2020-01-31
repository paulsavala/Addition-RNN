from models.seq2seq import SimpleSeq2Seq
from utils.integers import char_to_int_map, input_seq_length, target_seq_length
from utils.common import reverse_dict
from utils.training import format_targets
from utils.prediction import pprint_metrics
from data_gen.integer_addition import generate_samples

from tensorflow.keras.callbacks import ModelCheckpoint

from pathlib import Path
from datetime import datetime


class Config:
    n_terms = 3
    n_digits = 3
    train_size = 5 * 10**4
    validation_split = 0.1
    test_size = 10**3
    epochs = 200


class Mappings:
    char_to_int = char_to_int_map()
    int_to_char = reverse_dict(char_to_int)


X_train, y_train = generate_samples(n_samples=Config.train_size,
                                    n_terms=Config.n_terms,
                                    n_digits=Config.n_digits,
                                    int_encoder=Mappings.char_to_int,
                                    one_hot=True)
X_test, y_test = generate_samples(n_samples=Config.test_size,
                                  n_terms=Config.n_terms,
                                  n_digits=Config.n_digits,
                                  int_encoder=Mappings.char_to_int,
                                  one_hot=True)


if __name__ == '__main__':
    # Create the callbacks
    # Create a callback that saves the model's weights
    log_dir = Path('experiments/basic_addition/logs/')
    log_dir.mkdir(parents=True, exist_ok=True)

    current_dt_str = datetime.now().strftime("%M_%d_%Y__%H_%M_%S")

    checkpoint_dir = log_dir / Path('checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / Path(f'{current_dt_str}.ckpt')
    cp_callback = ModelCheckpoint(filepath=checkpoint_path.as_posix(),
                                  save_weights_only=True,
                                  verbose=1)

    # Define the model
    model_class = SimpleSeq2Seq(encoder_units=128,
                          batch_size=128,
                          input_seq_length=input_seq_length(Config.n_terms, Config.n_digits),
                          target_seq_length=target_seq_length(Config.n_terms, Config.n_digits),
                          vocab_size=len(Mappings.char_to_int)
                          )
    model = model_class.model()

    # Format the input and output
    input_train_target, output_train_target =  format_targets(y_train)
    model.fit([X_train, input_train_target],
              output_train_target,
              epochs=Config.epochs,
              validation_split=Config.validation_split,
              callbacks=[cp_callback])

    # Evaluate on the test set
    input_test_target, output_test_target = format_targets(y_test)
    test_metrics = model.evaluate([X_test, input_test_target], output_test_target, verbose=0)
    pprint_metrics(test_metrics, model.metrics_names)