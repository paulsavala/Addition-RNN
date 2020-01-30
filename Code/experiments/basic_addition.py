from models.seq2seq import SimpleSeq2Seq
from utils.integers import char_to_int_map, input_seq_length, target_seq_length
from utils.common import reverse_dict
from utils.prediction import pprint_metrics
from data_gen.integer_addition import generate_samples


class Config:
    n_terms = 2
    n_digits = 2
    train_size = 5 * 10**3
    validation_split = 0.1
    test_size = 10**3
    epochs = 5


class Mappings:
    char_to_int = char_to_int_map()
    int_to_char = reverse_dict(char_to_int)


X_train, y_train = generate_samples(n_samples=Config.train_size, int_encoder=Mappings.char_to_int, one_hot=True)
X_test, y_test = generate_samples(n_samples=Config.test_size, int_encoder=Mappings.char_to_int, one_hot=True)


if __name__ == '__main__':
    model_class = SimpleSeq2Seq(encoder_units=1,
                          decoder_units=1,
                          batch_size=64,
                          input_seq_length=input_seq_length(Config.n_terms, Config.n_digits),
                          target_seq_length=target_seq_length(Config.n_terms, Config.n_digits),
                          vocab_size=len(Mappings.char_to_int)
                          )
    model = model_class.model()

    input_target = y_train[:, :-1, :]
    output_target = y_train[:, 1:, :]
    model.fit([X_train, input_target], output_target, epochs=Config.epochs, validation_split=Config.validation_split)

    test_metrics = model.evaluate([X_test, y_test[:, :-1, :]], y_test[:, 1:, :], verbose=0)
    pprint_metrics(test_metrics, model.metrics_names)