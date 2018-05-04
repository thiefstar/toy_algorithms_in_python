# -*- coding:utf-8 -*-

import model_config
import data_utils
from bilstm_crf import BiLSTM_CRF_Model
import random

def main():

    config = model_config.Config()

    BATCH_SIZE = config.batch_size

    data = data_utils.read_data(config.corpus_path)

    data_utils.build_vocab(config.corpus_path, config.vocab_path)
    vocab = data_utils.read_dictionary()
    # random.shuffle(data)
    train_data = data[BATCH_SIZE:]
    test_data = data[:BATCH_SIZE]


    model = BiLSTM_CRF_Model(
        config=config,
        vocab=vocab,
        tag2label=data_utils.tag2label
    )

    model.build_graph()


    model.fit(train_data,
                val_data=test_data)


if __name__ == '__main__':
    main()