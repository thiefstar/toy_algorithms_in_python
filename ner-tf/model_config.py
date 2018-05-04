# -*- coding:utf-8 -*-

import os

class Config():

    def __init__(self):
        self.num_epochs = 80
        # Batch Size
        self.batch_size = 64
        # RNN Size
        self.num_units = 128
        # Embedding Size
        self.embedding_size = 64
        # Learning Rate
        self.learning_rate = 0.001
        # dropout keep prob
        self.dropout = 0.8
        # use CRF
        self.CRF = True
        # shuffle train data
        self.shuffle = False
        # dispaly step
        self.display_step = 128
        # 过滤低频word
        self.min_count = 0
        # save between training
        self.n_step_to_save = 4000  # or None  # 选一个更好的?
        # Path & File
        # 训练语料
        self.corpus_path = "data_path/train_data"
        # save path
        self.save_path = "save/model.ckpt"
        # summary_path
        self.summary_path = "save/summary"
        # word2idx pkl file
        self.vocab_path = "save/vocab.pkl"
