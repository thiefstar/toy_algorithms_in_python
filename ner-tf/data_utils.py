# -*- coding:utf-8 -*-

import model_config
import sys, pickle, os, random
import numpy as np

## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6,
             # "<EOS>": 7
             }


config = model_config.Config()


def read_data(path):
    """ char/tag a line and split sent with space line """
    data = []
    with open(path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data

def build_vocab(corpus_path, vocab_path, min_count=0):
    """ save  word2idx (only X, Y is just tag), not consider en text analysis"""
    specials = ['<PAD>', '<UNK>']
    vocabulary = {}
    data = read_data(corpus_path)
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):  # isalpha对中文也是True
                word = '<ENG>'
            vocabulary[word] = vocabulary.get(word, 0) + 1
    if min_count > 1:
        word_filter = [wd for wd, freq in vocabulary.items() if freq < min_count and wd != '<NUM>' and wd != '<ENG>']
        for word in word_filter:
            del vocabulary[word]


    word2idx = {word: idx for idx, word in enumerate(specials + list(vocabulary.keys()))}
    # idx2word = {idx: word for word, idx in word2idx.items()}

    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2idx, fw)


def read_dictionary():
    """ get the vocab """

    vocab_path = config.vocab_path
    if not os.path.exists(vocab_path):
        print('词汇表文件不存在，重新生成词汇表文件并存入：{}'.format(vocab_path))
        if not os.path.exists(config.corpus_path):
            raise Exception("未找到语料，请配置model_config.py中的corpus_path")
        build_vocab(config.corpus_path, vocab_path, config.min_count)
    with open(vocab_path, 'rb') as fr:
        word2idx = pickle.load(fr)
    print('vocab_size:', len(word2idx))
    return word2idx


def sent2idx(sent, word2idx):
    """ preprocess and 2idx"""
    word_indices = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'

        word_indices.append(word2idx.get(word, word2idx['<UNK>']))
    return word_indices


def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens
# end method pad_sentence_batch


def get_batches(data, batch_size, vocab, tag2label, shuffle=False):
    """
    :param data:  (sent, tags) list
    :param batch_size:
    :param vocab: word2idx dict
    :param tag2label:
    :param shuffle:
    :return: padded_seqs_batch,padded_label_seqs_batch, and len
    """

    if shuffle:
        random.shuffle(data)

    for i in range(0, len(data) - len(data) % batch_size, batch_size):
        data_batch = data[i : i + batch_size]
        seqs_batch = [sent2idx(seq, vocab) for seq, tags in data_batch]
        label_seqs_batch =[[tag2label[tag] for tag in tags] for seq, tags in data_batch]
        padded_X_batch, X_batch_lens = pad_sentence_batch(seqs_batch, vocab['<PAD>'])
        padded_Y_batch, Y_batch_lens = pad_sentence_batch(label_seqs_batch, tag2label['O'])
        yield (np.array(padded_X_batch),
               np.array(padded_Y_batch),
               X_batch_lens,
               Y_batch_lens)
# end method next_batch
