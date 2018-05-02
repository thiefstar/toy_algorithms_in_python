# -*- coding:utf-8 -*-

import os

class Config():

    def __init__(self):
        self.num_epochs = 10
        # Batch Size
        self.batch_size = 64
        # RNN Size
        self.num_units = 128
        # Number of Layers
        self.num_layers = 1
        # Embedding Size
        self.encoding_embedding_size = 128
        self.decoding_embedding_size = 128
        # Learning Rate
        self.learning_rate = 0.001
        self.display_step = 20
        # Path & File
        self.path = "duilian"
        self.source_fn = "data/train.enc"
        self.target_fn = "data/train.dec"
        self.source_path = os.path.join(self.path, self.source_fn)
        self.target_path = os.path.join(self.path, self.target_fn)
        # save path
        self.save_path = "save/model.ckpt"

        # processed_data
        self.PREPROCESS_DATA = "save/preprocess.pkl"
