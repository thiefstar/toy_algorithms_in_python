# -*- coding:utf-8 -*-

import model_config
import data_utils
from bilstm_crf import BiLSTM_CRF_Model


def main():

    config = model_config.Config()

    vocab = data_utils.read_dictionary()

    print(len(vocab))

    model = BiLSTM_CRF_Model(
        config=config,
        vocab=vocab,
        tag2label=data_utils.tag2label
    )

    model.build_graph()

    model.load_model(config.save_path)

    model.infer('首先是这一天，并且是访问百度的日志中的IP取出来，逐个写入到一个大文件中。')
    model.infer('中共中央致中国致公党十一大的贺词')
    model.infer('在中国致公党第十一次全国代表大会隆重召开之际，中国共产党中央委员会谨向大会表示热烈的祝贺，向致公党的同志们致以亲切的问候！')
    model.infer('这次代表大会是在中国改革开放和社会主义现代化建设发展的关键时刻召开的历史性会议。')
    model.infer('当前，在中共十五大精神的指引下，在以江泽民同志为核心的中共中央领导下，全党和全国各族人民正高举邓小平理论伟大旗帜，同心同德，团结奋斗，沿着建设有中国特色的社会主义道路阔步前进。')



def predict(source_sentences):
    """
    :param source_sentences: a list of sentences to predict
    :return: result list
    """
    pass


if __name__ == '__main__':
    main()