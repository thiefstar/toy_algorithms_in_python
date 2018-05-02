# -*- coding:utf-8 -*-

import os

class Config():

    def __init__(self):
        self.num_epochs = 10
        # Batch Size
        self.batch_size = 32
        # RNN Size
        self.num_units = 128
        # Number of Layers
        self.num_layers = 2
        # Embedding Size
        self.encoding_embedding_size = 64
        self.decoding_embedding_size = 64
        # Learning Rate
        self.learning_rate = 0.001
        self.display_step = 32
        # Path & File
        # self.path = "text-data/zh-en"
        # self.source_fn = "chinese.raw.txt"
        # self.target_fn = "english.raw.txt"
        self.source_path = os.path.join(self.path, self.source_fn)
        self.target_path = os.path.join(self.path, self.target_fn)
        # save path
        self.save_path = "save/model.ckpt"
        # processed_data
        self.preprocess_fn = "save/preprocess.pkl"
