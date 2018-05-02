# -*- coding:utf-8 -*-

import model_config
import data_utils
from seq2seq_attn import Seq2Seq


config = model_config.Config()

def main():
    _, (X_char2idx, Y_char2idx), (X_idx2char, Y_idx2char) = \
        data_utils.load_preprocess(config.source_path, config.target_path)

    model = Seq2Seq(
        rnn_size = config.num_units,
        n_layers = config.num_layers,
        X_word2idx = X_char2idx,
        encoder_embedding_dim = config.encoding_embedding_size,
        Y_word2idx = Y_char2idx,
        decoder_embedding_dim = config.decoding_embedding_size,
        load_path=config.save_path
    )

    model.infer('今朝有酒今朝醉', X_idx2char, Y_idx2char)
    # model.infer('你好', X_idx2char, Y_idx2char)
    model.infer('雪消狮子瘦', X_idx2char, Y_idx2char)
    # model.infer('晚上吃什么', X_idx2char, Y_idx2char)
    model.infer('生员里长，打里长不打生员。', X_idx2char, Y_idx2char)
    # model.infer('我没什么意见', X_idx2char, Y_idx2char)



def predict(source_sentences):
    """
    :param source_sentences: a list of sentences to predict
    :return: result list
    """
    pass


if __name__ == '__main__':
    main()