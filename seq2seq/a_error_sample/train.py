# -*- coding:utf-8 -*-

import tensorflow as tf
import config
from my_seq2seq_model import Seq2Seq_Model
import data_utils

# config
config = config.Config()


def train():

    if not tf.gfile.Exists(config.PREPROCESS_DATA):
        print('预处理文件不存在，重新生成预处理文件并存入：{}'.format(config.PREPROCESS_DATA))
        data_utils.preprocess_and_save_data(config.source_path, config.target_path)

    (source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = data_utils.load_preprocess()

    print("训练数据加载成功")


    train_graph = tf.Graph()
    with train_graph.as_default():

        model = Seq2Seq_Model(
            num_units=config.num_units, 
            # keep_prob,
            batch_size=config.batch_size,
            source_vocab_size=len(source_vocab_to_int), 
            target_vocab_size=len(target_vocab_to_int),
            encoding_embedding_size=config.encoding_embedding_size, 
            decoding_embedding_size=config.decoding_embedding_size, 
            target_vocab_to_int=target_vocab_to_int,
            mode='train')

        model.build_model()

        # Split data to training and validation sets
        batch_size = config.batch_size
        train_source = source_int_text[batch_size:]
        train_target = target_int_text[batch_size:]
        valid_source = source_int_text[:batch_size]
        valid_target = target_int_text[:batch_size]
        (valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths) = next(
            data_utils.get_batches(valid_source,
                        valid_target,
                        batch_size,
                        source_vocab_to_int['<PAD>'],
                        target_vocab_to_int['<PAD>']))

        with tf.Session(graph=train_graph, config=tf.ConfigProto(device_count={'GPU': 0})) as sess:

            sess.run(tf.global_variables_initializer())

            for epoch_i in range(1, config.num_epochs+1):
                for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in \
                    enumerate(data_utils.get_batches(train_source, train_target, config.batch_size,
                                source_vocab_to_int['<PAD>'],
                                target_vocab_to_int['<PAD>'])):

                    train_loss = model.train(sess,
                                                         source_batch,
                                                         target_batch,
                                                         sources_lengths,
                                                         targets_lengths,
                                                         config.learning_rate)

                    if batch_i % config.display_step == 0 and batch_i > 0:
                        valid_loss = model.eval(sess,
                                                valid_sources_batch,
                                                valid_targets_batch,
                                                valid_sources_lengths,
                                                valid_targets_lengths,)

                        print('Epoch {:>3} Batch {:>4}/{} - Loss: {:>6.4f}, Valid Loss: {:6.4f}'
                                .format(epoch_i, batch_i, len(source_int_text) // batch_size,
                                        train_loss, valid_loss))
#  Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f},  | , train_acc, valid_acc

            # Save Model
            saver = tf.train.Saver()
            saver.save(sess, config.save_path)
            print('Model Trained and Saved')


if __name__ == '__main__':
    train()