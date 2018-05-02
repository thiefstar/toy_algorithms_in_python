# encoding=utf-8
import tensorflow as tf
from my_seq2seq_model import Seq2Seq_Model
import config
import data_utils

# config
config = config.Config()

load_path = config.save_path


def sentence_to_seq(sentence, vocab_to_int):
    '''
    将测试句子转换成输入id
    '''
    sent = sentence.lower()
    sent_to_id = [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in sent]
    return sent_to_id


def restore_model(source_vocab_to_int, target_vocab_to_int):

    sess = tf.Session()
    model = Seq2Seq_Model(
        num_units=config.num_units,
        batch_size=config.batch_size,
        source_vocab_size=len(source_vocab_to_int),
        target_vocab_size=len(target_vocab_to_int),
        encoding_embedding_size=config.encoding_embedding_size,
        decoding_embedding_size=config.decoding_embedding_size,
        target_vocab_to_int=target_vocab_to_int,
        mode='inference')

    model.build_model()

    saver = tf.train.Saver()
    saver.restore(sess, config.save_path)

    return model, sess


def test():

    _, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = data_utils.load_preprocess()

    model, sess = restore_model(source_vocab_to_int, target_vocab_to_int)

    logits = model.inference_logits

    def test_sentence(translate_sentence):

        translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

        translate_logits = sess.run(logits, feed_dict={model.input_data: [translate_sentence[::-1]] * config.batch_size,
                                           model.source_sequence_length: [len(translate_sentence)] * config.batch_size})[0]

        print('Input')
        print('  Word Ids:      {}'.format([i for i in translate_sentence]))
        print('  source Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

        print('\nPrediction')
        print('  Word Ids:      {}'.format([i for i in translate_logits[:-1]]))
        print('  target Words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits[:-1]])))


    test_sentence('今朝有酒今朝醉')
    test_sentence('雪消狮子瘦')
    test_sentence('生员里长，打里长不打生员。')
    test_sentence('山茶')



if __name__ == '__main__':
    test()