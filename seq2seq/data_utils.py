# -*- coding:utf-8 -*-

import os
import model_config
import pickle

config = model_config.Config()

def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
# end function read_data


def build_map(data, space_tokenizer=False):
    """ todo: space_tokenizer: 针对分词的情况, 但实际上english tokenize还要再复杂一些"""
    specials = ['<GO>',  '<EOS>', '<PAD>', '<UNK>']
    if space_tokenizer:
        chars = list(set([word for line in data.split('\n') for word in line.split(' ')]))
    else:
        chars = list(set([word for line in data.split('\n') for word in line]))
    idx2char = {idx: char for idx, char in enumerate(specials + chars)}
    char2idx = {char: idx for idx, char in idx2char.items()}
    return idx2char, char2idx
# end function build_map

def preprocess_data(source_path, target_path):
    X_data = read_data(source_path)
    Y_data = read_data(target_path)

    X_idx2char, X_char2idx = build_map(X_data)
    Y_idx2char, Y_char2idx = build_map(Y_data)

    x_unk = X_char2idx['<UNK>']
    y_unk = Y_char2idx['<UNK>']
    y_eos = Y_char2idx['<EOS>']

    X_indices = [[X_char2idx.get(char, x_unk) for char in line] for line in X_data.split('\n')]
    Y_indices = [[Y_char2idx.get(char, y_unk) for char in line] + [y_eos] for line in Y_data.split('\n')]

    return X_indices, Y_indices, X_char2idx, Y_char2idx, X_idx2char, Y_idx2char
# end function preprocess_data

def preprocess_and_save_data(source_path, target_path):

    preprocess_fn = config.preprocess_fn
    X_indices, Y_indices, X_char2idx, Y_char2idx, X_idx2char, Y_idx2char = \
        preprocess_data(source_path, target_path)

    dir = os.sep.join(preprocess_fn.split(os.sep)[:-1])
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(preprocess_fn, 'wb') as out_file:
        pickle.dump((
            (X_indices, Y_indices),
            (X_char2idx, Y_char2idx),
            (X_idx2char, Y_idx2char)), out_file)

def load_preprocess(source_path, target_path):

    preprocess_fn = config.preprocess_fn
    if not os.path.exists(preprocess_fn):
        print('预处理文件不存在，重新生成预处理文件并存入：{}'.format(preprocess_fn))
        preprocess_and_save_data(source_path, target_path)
    with open(preprocess_fn, mode='rb') as in_file:
        return pickle.load(in_file)

