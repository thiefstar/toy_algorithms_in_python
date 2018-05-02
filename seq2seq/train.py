# -*- coding:utf-8 -*-

import model_config
import data_utils
from seq2seq_attn import Seq2Seq

def main():

    config = model_config.Config()

    BATCH_SIZE = config.batch_size

    (X_indices, Y_indices), (X_char2idx, Y_char2idx), (X_idx2char, Y_idx2char) = \
        data_utils.load_preprocess(config.source_path, config.target_path)

    X_train = X_indices[BATCH_SIZE:]
    Y_train = Y_indices[BATCH_SIZE:]
    X_test = X_indices[:BATCH_SIZE]
    Y_test = Y_indices[:BATCH_SIZE]


    model = Seq2Seq(
        rnn_size = config.num_units,
        n_layers = config.num_layers,
        X_word2idx = X_char2idx,
        encoder_embedding_dim = config.encoding_embedding_size,
        Y_word2idx = Y_char2idx,
        decoder_embedding_dim = config.decoding_embedding_size,
    )


    model.fit(X_train, Y_train,
                val_data=(X_test, Y_test),
                batch_size=BATCH_SIZE,
                n_epoch=config.num_epochs,
                display_step=config.display_step)

    model.save_model(config.save_path)


if __name__ == '__main__':
    main()