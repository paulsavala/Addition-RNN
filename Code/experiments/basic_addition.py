from models.seq2seq import SimpleSeq2Seq
from utils.integers import char_to_int_map, input_seq_length, target_seq_length
from utils.common import reverse_dict


class Config:
    n_terms = 2
    n_digits = 2


class Mappings:
    char_to_int = char_to_int_map()
    int_to_char = reverse_dict(char_to_int)


if __name__ == '__main__':
    model = SimpleSeq2Seq(encoder_units=1,
                          decoder_units=1,
                          batch_size=64,
                          input_seq_length=input_seq_length(Config.n_terms, Config.n_digits),
                          target_seq_length=target_seq_length(Config.n_terms, Config.n_digits),
                          vocab_size=len(Mappings.char_to_int)
                          )