# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import os

# todo: 增加正确率的计算

class BaseSeq2Seq:

    def pad_sentence_batch(self, sentence_batch, pad_int):
        padded_seqs = []
        seq_lens = []
        max_sentence_len = max([len(sentence) for sentence in sentence_batch])
        for sentence in sentence_batch:
            padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
            seq_lens.append(len(sentence))
        return padded_seqs, seq_lens
    # end method pad_sentence_batch


    def next_batch(self, X, Y, batch_size):
        for i in range(0, len(X) - len(X) % batch_size, batch_size):
            X_batch = X[i : i + batch_size]
            Y_batch = Y[i : i + batch_size]
            padded_X_batch, X_batch_lens = self.pad_sentence_batch(X_batch, self._x_pad)
            padded_Y_batch, Y_batch_lens = self.pad_sentence_batch(Y_batch, self._y_pad)
            yield (np.array(padded_X_batch),
                   np.array(padded_Y_batch),
                   X_batch_lens,
                   Y_batch_lens)
    # end method next_batch


    def fit(self, X_train, Y_train, val_data, n_epoch=20, display_step=50, batch_size=128):
        X_test, Y_test = val_data
        X_test_batch, Y_test_batch, X_test_batch_lens, Y_test_batch_lens = next(
        self.next_batch(X_test, Y_test, batch_size))

        # self.sess.run(tf.global_variables_initializer())
        for epoch in range(1, n_epoch+1):
            for local_step, (X_train_batch, Y_train_batch, X_train_batch_lens, Y_train_batch_lens) in enumerate(
                self.next_batch(X_train, Y_train, batch_size)):
                _, loss = self.sess.run([self.train_op, self.loss], {self.X: X_train_batch,
                                                                     self.Y: Y_train_batch,
                                                                     self.X_seq_len: X_train_batch_lens,
                                                                     self.Y_seq_len: Y_train_batch_lens})
                if local_step % display_step == 0:
                    val_loss = self.sess.run(self.loss, {self.X: X_test_batch,
                                                         self.Y: Y_test_batch,
                                                         self.X_seq_len: X_test_batch_lens,
                                                         self.Y_seq_len: Y_test_batch_lens})
                    print("Epoch %d/%d | Batch %d/%d | train_loss: %.3f | test_loss: %.3f"
                        % (epoch, n_epoch, local_step, len(X_train)//batch_size, loss, val_loss))
    # end method fit


    def infer(self, input_word, X_idx2word, Y_idx2word, batch_size=128):        
        input_indices = [self.X_word2idx.get(char, self._x_unk) for char in input_word]
        out_indices = self.sess.run(self.predicting_ids, {
            self.X: [input_indices] * batch_size,
            self.X_seq_len: [len(input_indices)] * batch_size})[0]
        
        print('IN: {}'.format(' '.join([X_idx2word[i] for i in input_indices])))
        print('OUT: {}'.format(' '.join([Y_idx2word[i] for i in out_indices])))
        print()
    # end method infer

    def load_model(self, load_path):
        with self.graph.as_default():
            if load_path and  os.path.exists(load_path + '.meta'):
                self.build_graph()
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, load_path)
                print("Model Loaded.")
            else:
                self.build_graph()
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver()

    def save_model(self, save_path):
        """use after fit"""
        self.saver.save(self.sess, save_path)
        print('Model Saved.')


    def register_symbols(self):
        self._x_go = self.X_word2idx['<GO>']
        self._x_eos = self.X_word2idx['<EOS>']
        self._x_pad = self.X_word2idx['<PAD>']
        self._x_unk = self.X_word2idx['<UNK>']

        self._y_go = self.Y_word2idx['<GO>']
        self._y_eos = self.Y_word2idx['<EOS>']
        self._y_pad = self.Y_word2idx['<PAD>']
        self._y_unk = self.Y_word2idx['<UNK>']
    # end method add_symbols
# end class
