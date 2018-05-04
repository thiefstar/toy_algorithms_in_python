# -*- coding:utf-8 -*-

# basic
# github.com/Determined22/zh-NER-TF
# github.com/rockingdingo/deepnlp/tree/master/deepnlp/ner
# https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/nlp-models/tensorflow/tf-data-api/pos_birnn_crf_test.ipynb (pos)

import numpy as np
import os, time
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
from data_utils import get_batches, sent2idx, pad_sentence_batch

class BiLSTM_CRF_Model(object):

    def __init__(self, config, vocab, tag2label):
        self.batch_size = config.batch_size
        self.epoch_num = config.num_epochs
        self.hidden_dim = config.num_units
        self.dropout_keep_prod = config.dropout
        self.lr = config.learning_rate
        self.CRF = config.CRF
        self.embedding_size = config.embedding_size
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.tag2label = tag2label
        self.label2tag = {label: tag for tag, label in tag2label.items()}
        self.num_tags = len(tag2label)
        self.n_step_to_save = config.n_step_to_save
        self.display_step = config.display_step
        self.summary_path = config.summary_path
        self.model_path = config.save_path
        self.sess = None

    def build_graph(self):
        self.add_placeholders()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()


    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        # self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")


    def init_op(self):
        self.init_op = tf.global_variables_initializer()


    def biLSTM_layer_op(self):
        with tf.variable_scope("words"):
            word_embeddings = tf.Variable(tf.random_uniform([len(self.vocab), self.embedding_size]), dtype=tf.float32, name="word_embeddings")
            embed_input = tf.nn.embedding_lookup(word_embeddings, self.word_ids, name="embed_input")

        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=embed_input,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)  # 观察一下
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])


    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)


    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                   tag_indices=self.labels,
                                                                   sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)


    def trainstep_op(self):
        """ """
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            # optimizer : SGD, RMSProp, Momentum, Adam..
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

            # 梯度裁剪
            grads_and_vars = optimizer.compute_gradients(self.loss)
            #  -self.clip_grad, self.clip_grad  裁减范围
            grads_and_vars_clip = [[tf.clip_by_value(grad, -1., 1.), var] for grad, var in grads_and_vars if grad is not None]
            self.train_op = optimizer.apply_gradients(grads_and_vars_clip, global_step=self.global_step)  # will plus 1 every batch?

    def fit(self, train_data, val_data):

        train_data_len = len(train_data)

        saver = tf.train.Saver(tf.global_variables())

        self.sess = tf.Session()
        self.sess.run(self.init_op)
        self.add_summary(self.sess)

        (X_val_batch, Y_val_batch, X_val_batch_lens, Y_val_batch_lens) = \
            next(get_batches(val_data, self.batch_size, self.vocab, self.tag2label))

        for epoch in range(1, self.epoch_num + 1):

            for local_step, (X_train_batch, Y_train_batch, X_train_batch_lens, Y_train_batch_lens) in enumerate(
                    get_batches(train_data, self.batch_size, self.vocab, self.tag2label)):
                _, loss, summary, step_num = self.sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                                    {self.word_ids: X_train_batch,
                                                                     self.labels: Y_train_batch,
                                                                     self.sequence_lengths: X_train_batch_lens,
                                                                         self.dropout_pl: self.dropout_keep_prod})
                if local_step % self.display_step == 0:
                    val_loss = self.sess.run(self.loss, {self.word_ids: X_val_batch,
                                                    self.labels: Y_val_batch,
                                                     self.sequence_lengths: X_val_batch_lens,
                                                     self.dropout_pl: self.dropout_keep_prod})
                    print("Epoch %d/%d | Batch %d/%d | train_loss: %.3f | val_loss: %.3f"
                        % (epoch, self.epoch_num, local_step, len(train_data)//self.batch_size, loss, val_loss))

                self.file_writer.add_summary(summary, step_num)

                if self.n_step_to_save and step_num % self.n_step_to_save == 0 and step_num != 0:  # every n step
                    saver.save(self.sess, self.model_path, global_step=step_num)
                    print("Model Saved... at time step " + str(step_num))

        saver.save(self.sess, self.model_path)
        print("Model Saved.")
        self.sess.close()
    # end method fit

    def infer(self, input_word):
        """ predict one seq """

        input_indices, input_indices_lens = pad_sentence_batch([sent2idx(input_word, self.vocab)] * self.batch_size, pad_int=self.vocab['<PAD>'])

        if self.CRF:
            logits, transition_params = self.sess.run([self.logits, self.transition_params],
                                                      {self.word_ids: input_indices,
                                                       self.sequence_lengths: input_indices_lens,
                                                        self.dropout_pl: 1.0})

            logit = logits[0]
            seq_len = input_indices_lens[0]
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)

            labels = viterbi_seq

        else:
            label_list = self.sess.run(self.labels_softmax_, feed_dict={self.word_ids: input_indices,
                                                                        self.sequence_lengths: input_indices_lens,
                                                                        self.dropout_pl: 1.0})
            labels = label_list[0]
            seq_len = input_indices_lens[0]

        print(len(input_indices))

        print('IN: {}'.format('\t'.join(['{:^5}'.format(i) for i in input_word])))
        print('OUT:{}'.format('\t'.join(['{:^5}'.format(self.label2tag[i]) for i in labels[:seq_len]])))
        print()

        return labels[:seq_len]
        # end method infer


    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def load_model(self, load_path):
        if os.path.exists(load_path + '.meta'):
            self.sess = tf.Session()
            self.sess.run(self.init_op)

            saver = tf.train.Saver()
            saver.restore(self.sess, self.model_path)
            print("Model Loaded.")
        else:
            raise Exception("model file: '{}' is not exist".format(load_path))

    # def save_model(self, save_path):
    #     pass

    def evaluate(self):
        """ evaluate test data """
        pass
