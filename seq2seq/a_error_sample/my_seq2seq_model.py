# -*- coding:utf-8 -*-

# wanna do
# basic seq2seq + attention (单层LSTM)
# 任务为一个zh-cn机器翻译模型，尽量通用明了
# 数据使用的是小牛翻译的 zh-en sample

# base
# https://github.com/tensorflow/tensorflow/blob/64edd34ce69b4a8033af5d217cb8894105297d8a/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py
# http://www.hankcs.com/nlp/cs224n-mt-lstm-gru.html
# https://github.com/zhedongzheng/finch/blob/master/nlp-models/tensorflow/seq2seq_attn.py
# https://github.com/tensorflow/nmt/blob/master/nmt/model.py  (上一个有点模仿这个，比较清楚)
# https://github.com/wb14123/seq2seq-couplet (对联)
# https://github.com/ByzenMa/deepnlp-models (当前文件结构类似于这个, 但加入attention引起的VarScope矛盾似乎无法解决)
# https://github.com/whikwon/seq2seq-attention/blob/083f13f81f709afda4097570a897002862dbf46c/chatbot_toycode/train.py (参考这个做了修改)

# todo: bucket data
# todo: 数据清洗
# todo: acc 计算
# todo: 问题多出在attention上，再多加了解
# 整体结构优化，再清楚明了一些
# 目前8G内存直接耗尽，暂不能验证正确性(很大概率是有问题)

# words:
# logits
# inference
#

from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, DropoutWrapper
import tensorflow as tf
import numpy as np

""" 
http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
BasicRNNCell – A vanilla RNN cell.
GRUCell – A Gated Recurrent Unit cell.
BasicLSTMCell – An LSTM cell based on Recurrent Neural Network Regularization. No peephole connection or cell clipping.
LSTMCell – A more complex LSTM cell that allows for optional peephole connections and cell clipping.
MultiRNNCell – A wrapper to combine multiple cells into a multi-layer cell.
DropoutWrapper – A wrapper to add dropout to input and/or output connections of a cell.
"""

class Seq2Seq_Model(object):

    def __init__(self, num_units, batch_size,
                source_vocab_size, target_vocab_size,
                encoding_embedding_size, decoding_embedding_size, target_vocab_to_int, mode):
        self.num_units = num_units  # hidden layer size
        self.batch_size = batch_size
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.encoding_embedding_size = encoding_embedding_size
        self.decoding_embedding_size = decoding_embedding_size
        self.target_vocab_to_int = target_vocab_to_int
        self.mode = mode  # train or inference

    def build_model(self):
        self.add_placeholders()
        self.build_encoder_layer()
        self.build_decoder_layer()

    def add_placeholders(self):
        self.input_data = tf.placeholder(tf.int32, shape=[None, None], name='input')
        self.input_data = tf.reverse(self.input_data, [-1])
        self.target_data = tf.placeholder(tf.int32, shape=[None, None], name='target')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        # self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.target_sequence_length = tf.placeholder(tf.int32, shape=(None,), name='target_sequence_length')

        self.source_sequence_length = tf.placeholder(tf.int32, shape=(None,), name='source_sequence_length')

    def build_encoder_layer(self):
        """
        编码, 使用单层的LSTM
        """
        # 放到哪里好? self.input_data = tf.reverse(self.input_data, [-1]),  # trick: reverse the input text

        enc_embed_input = tf.contrib.layers.embed_sequence(self.input_data,
                                                            self.source_vocab_size,
                                                            self.encoding_embedding_size)
        # Build RNN cell
        enc_cell = tf.contrib.rnn.LSTMCell(self.num_units, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        # Run Dynamic RNN
        #   encoder_outputs: [max_time, batch_size, num_units]
        #   encoder_state: [batch_size, num_units]

        print(self.input_data)
        print(self.source_vocab_size)
        print(self.encoding_embedding_size)
        print(enc_embed_input)

        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
            enc_cell, enc_embed_input,
            sequence_length=self.source_sequence_length, dtype=tf.float32)

        # 在Train的时候，需要对需要encoder的数据进行修改
        # if self.mode == 'train':
        #     pass

    def build_decoder_layer(self):
        """
        """
        # 将解码数据变换成以'<GO>'开始，并去除末尾的'<EOS>', 用于decoder的输入
        go_idx = self.target_vocab_to_int['<GO>']
        tiled_go_idx = tf.cast(tf.reshape(tf.tile(tf.constant([go_idx], dtype=tf.int32),
                [self.batch_size]), shape=[self.batch_size, -1]), tf.int32)
        dec_input =  tf.concat([tiled_go_idx, self.target_data], axis=1)[:, :-1]

        dec_cell = tf.contrib.rnn.LSTMCell(self.num_units, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))

        # Projection, output layer
        projection_layer = layers_core.Dense(self.target_vocab_size, use_bias=False)


        # attention_mechanism = tf.contrib.seq2seq
        # attn_cell = tf.contrib.seq2seq.DynamicAttentionWrapper
        # multi_cell = MultiRNNCell([attn_cell, top_cell])  ?
        # attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])  #tf nmt

        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units = self.num_units,  # 需要 x2 吗?
            memory = self.encoder_outputs,
            memory_sequence_length = self.source_sequence_length)

        # attention wrapper, attention 封装后的 RNNCell
        dec_cell = tf.contrib.seq2seq.AttentionWrapper(
            dec_cell, attention_mechanism, attention_layer_size=self.num_units)  # x2?

        decoder_initial_state = dec_cell.zero_state(
            self.batch_size, tf.float32).clone(cell_state=self.encoder_state)

        # dec_embeddings: 变量，在预测的时候...
        dec_embeddings = tf.Variable(tf.random_uniform([self.target_vocab_size, self.decoding_embedding_size]), name="decoder_embedding")


        if self.mode == 'train':
            dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

            # Helper : 再仔细了解一下?
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = dec_embed_input,
                sequence_length = self.target_sequence_length,
                time_major = False,
                name = 'training_helper')

            # basic decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=dec_cell,
                helper=training_helper,
                initial_state=decoder_initial_state,
                output_layer=projection_layer)

            self.max_target_sequence_length = tf.reduce_max(self.target_sequence_length, name='max_target_len')
            # Dynamic decoding
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder,
                impute_finished = True,
                #impute_finished ?? Boolean，为真时会拷贝最后一个时刻的状态并将输出置零，程序运行更稳定，使最终状态和输出具有正确的值，在反向传播时忽略最后一个完成步。但是会降低程序运行速度。
                maximum_iterations= self.max_target_sequence_length) #  tf.reduce_max(self.target_sequence_length) 或者需要将reduce_max放到这边来

            training_logits = tf.identity(
                training_decoder_output.rnn_output, name='logits')

            # LOSS
            masks = tf.sequence_mask(self.target_sequence_length, self.max_target_sequence_length, dtype=tf.float32,
                                 name='mask')
            self.cost = tf.contrib.seq2seq.sequence_loss(
                                logits=training_logits,
                                targets=self.target_data,
                                weights=masks)

            # # ACCUARY
            # self.acc = self.accuary(self.target_data, training_logits)

            # BACKWARD  主要的作用?  params 包括哪些， lr...
            params = tf.trainable_variables()
            gradients = tf.gradients(self.cost, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.train_op = tf.train.AdamOptimizer().apply_gradients(
                zip(clipped_gradients, params))

        # INFERENCE
        elif self.mode == 'inference':

            start_of_sequence_id = self.target_vocab_to_int['<GO>']
            end_of_sequence_id = self.target_vocab_to_int['<EOS>']
            start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), [self.batch_size])

            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding = dec_embeddings,
                start_tokens = start_tokens,
                end_token = end_of_sequence_id)

            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=dec_cell,
                helper=predicting_helper,
                initial_state=decoder_initial_state,
                output_layer=projection_layer)

            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = predicting_decoder,
                impute_finished = True,
                maximum_iterations = 2 * tf.reduce_max(self.source_sequence_length)  # ?
            )

            self.inference_logits = tf.identity(
                predicting_decoder_output.sample_id, name='predictions')


    # Train STEP, return train_loss
    def train(self, sess, input_data, target_data,
              source_sequence_length, target_sequence_length, lr):

        if self.mode.lower() != 'train':
            raise ValueError("train step can only be operated in train mode")

        input_feed = self.check_feeds(input_data, source_sequence_length,
                                      target_data, target_sequence_length, lr, decode=False)

        output_feed = [self.train_op, self.cost]
        outputs = sess.run(output_feed, input_feed)
        return outputs[1]  # loss


    # EVAL STEP, return val_loss
    def eval(self, sess, input_data, target_data,
              source_sequence_length, target_sequence_length):

        if self.mode.lower() != 'train':
            raise ValueError("train step can only be operated in train mode")

        input_feed = self.check_feeds(input_data, source_sequence_length,
                                      target_data, target_sequence_length, lr=None, decode=False)

        output_feed = [self.cost]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0]  # loss


    # 此处因无法确定pad的大小，没办法加进graph内, 把np换tf是否会有用
    # train_acc = model.accuary(target_batch, batch_train_logits)
    # valid_acc = model.accuary(valid_targets_batch, batch_valid_logits)
    def accuary(self, target, logits):
        max_seq = max(target.shape[1], logits.shape[1])
        print(max_seq, target.shape[1], logits.shape[1])
        if max_seq - target.shape[1]:
            target = np.pad(
                target,
                [(0, 0), (0, max_seq - target.shape[1])],
                'constant')
        if max_seq - logits.shape[1]:
            logits = np.pad(
                logits,
                [(0, 0), (0, max_seq - logits.shape[1])],
                'constant')
        return np.mean(np.equal(target, logits))


    def check_feeds(self, input_data, source_sequence_length,
                    target_data, target_sequence_length, lr, decode):
        input_batch_size = input_data.shape[0]
        if input_batch_size != source_sequence_length.shape[0]:
            raise ValueError("Encoder inputs and their lengths must be equal in their "
                             "batch_size, %d != %d" % (input_batch_size, source_sequence_length.shape[0]))

        if not decode:
            target_batch_size = target_data.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError("Encoder inputs and Decoder inputs must be equal in their "
                                 "batch_size, %d != %d" % (input_batch_size, target_batch_size))
            if target_batch_size != target_sequence_length.shape[0]:
                raise ValueError("Decoder targets and their lengths must be equal in their "
                                 "batch_size, %d != %d" % (target_batch_size, target_sequence_length.shape[0]))

        input_feed = {}

        input_feed[self.input_data.name] = input_data
        input_feed[self.source_sequence_length.name] = source_sequence_length

        if not decode:
            input_feed[self.target_data.name] = target_data
            input_feed[self.target_sequence_length.name] = target_sequence_length
            input_feed[self.lr.name] = lr

        return input_feed