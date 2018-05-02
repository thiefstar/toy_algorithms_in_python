# -*- coding:utf-8 -*-

import os
import numpy as np
import config
import pickle

PAD = 0
UNK = 1
GO = 2
EOS = 3
start_token = GO
end_token = EOS

config = config.Config()

PREPROCESS_DATA = config.PREPROCESS_DATA

def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8', errors='ignore') as f:
        data = f.read()
    return data

def extract_character_vocab(data, space_tokenizer=True):
    # 暂时没考虑取常用words
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

    if space_tokenizer:
        set_words = set([word for line in data.split('\n') for word in line.split(' ')])
    else:
        set_words = set([character for line in data.split('\n') for character in line])
    int_to_vocab = {word_i: word for word_i, word in enumerate(special_words + list(set_words))}
    vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int


def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, np.array(pad_source_lengths), np.array(pad_targets_lengths)


def text_to_ids(text, vocab_to_int, is_target=False, space_tokenizer=True):
    """
    :param vocab_to_int: vocab_to_int_dict
    :param is_target:如果是目标文本, 在最后添加'<EOS>'
    :param space_tokenizer: ''(表示按单个字分隔) or ' '(表示按空格分隔，指英语、分过词的汉语等等)
    """
    if space_tokenizer:
        sents = [" ".join(sent) for sent in text.split('\n')]
    else:
        sents = text.split('\n')
    if is_target:
        sents = [sent + ' <EOS>' for sent in sents]
        
    id_text = []
    for sent in sents:
        ids = [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in sent.split(' ') if len(word.strip()) > 0]  
        id_text.append(ids)
    return id_text

def preprocess_and_save_data(source_path, target_path):
    source_sentences = load_data(config.source_path)
    target_sentences = load_data(config.target_path)
    source_sentences = source_sentences.lower()
    target_sentences = target_sentences.lower()

    # process data
    # source_int_text: id text
    # if process char-level Chinese , use text_to_ids's param: space_tokenizer=False
    source_int_to_vocab, source_vocab_to_int = extract_character_vocab(source_sentences,
                                                                                  space_tokenizer=False)
    source_int_text = text_to_ids(source_sentences, source_vocab_to_int, space_tokenizer=False)

    target_int_to_vocab, target_vocab_to_int = extract_character_vocab(target_sentences,
                                                                                  space_tokenizer=False)
    target_int_text = text_to_ids(target_sentences, target_vocab_to_int, is_target=True,
                                             space_tokenizer=False)

    if not os.path.exists(PREPROCESS_DATA):
        os.mkdir(os.sep.join(PREPROCESS_DATA.split(os.sep)[:-1]))
    with open(PREPROCESS_DATA, 'wb') as out_file:
        pickle.dump((
            (source_int_text, target_int_text),
            (source_vocab_to_int, target_vocab_to_int),
            (source_int_to_vocab, target_int_to_vocab)), out_file)


def load_preprocess():
    with open(PREPROCESS_DATA, mode='rb') as in_file:
        return pickle.load(in_file)
