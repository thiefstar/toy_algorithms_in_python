# -*- coding:utf-8 -*-

# basic
# https://github.com/facebookresearch/fastText/blob/master/python/doc/examples/train_supervised.py
# blog.csdn.net/lxg0807/article/details/52960072

"""
train_supervised(input, lr=0.1, dim=100, ws=5, epoch=5, minCount=1, minCountLabel=0, minn=0, maxn=0, neg=5, wordNgrams=1, loss='softmax', bucket=2000000, thread=12, lrUpdateRate=100, t=0.0001, label='__label__', verbose=2, pretrainedVectors='')
    Train a supervised model and return a model object.
        
    input must be a filepath. The input text does not need to be tokenized
    as per the tokenize function, but it must be preprocessed and encoded
    as UTF-8. You might want to consult standard preprocessing scripts such
    as tokenizer.perl mentioned here: http://www.statmt.org/wmt07/baseline.html
    
    The input file must must contain at least one label per line. For an
    example consult the example datasets which are part of the fastText
    repository such as the dataset pulled by classification-example.sh.

train_unsupervised(input, model='skipgram', lr=0.05, dim=100, ws=5, epoch=5, minCount=5, minCountLabel=0, minn=3, maxn=6, neg=5, wordNgrams=1, loss='ns', bucket=2000000, thread=12, lrUpdateRate=100, t=0.0001, label='__label__', verbose=2, pretrainedVectors='')
    Train an unsupervised model and return a model object.
    
    input must be a filepath. The input text does not need to be tokenized
    as per the tokenize function, but it must be preprocessed and encoded
    as UTF-8. You might want to consult standard preprocessing scripts such
    as tokenizer.perl mentioned here: http://www.statmt.org/wmt07/baseline.html
    
    The input field must not contain any labels or use the specified label prefix
    unless it is ok for those words to be ignored. For an example consult the
    dataset pulled by the example script word-vector-example.sh, which is
    part of the fastText repository.

classifier.predict(text, k=1, threshold=0.0)
    Docstring:
    Given a string, get a list of labels and a list of
    corresponding probabilities. k controls the number
    of returned labels. A choice of 5, will return the 5
    most probable labels. By default this returns only
    the most likely label and probability. threshold filters
    the returned labels by a threshold on probability. A
    choice of 0.5 will return labels with at least 0.5
    probability. k and threshold will be applied together to
    determine the returned labels.

    This function assumes to be given
    a single line of text. We split words on whitespace (space,
    newline, tab, vertical tab) and the control characters carriage
    return, formfeed and the null character.

    If the model is not supervised, this function will throw a ValueError.

    If given a list of strings, it will return a list of results as usually
    received for a single line of text.

"""


import fastText
import jieba
import re
from opencc import OpenCC  

openCC = OpenCC('t2s')

POS_LABEL = '__label__pos'
NEG_LABEL = '__label__neg'

def train():
    # (先准备好训练语料)
    ftrain = 'reviews_fasttext_train.txt'
    ftest = 'reviews_fasttext_test.txt'

    # 训练模型 
    classifier = fastText.train_supervised(ftrain, label="__label__")
    classifier.save_model("reviews_fasttext.bin")

def predict(content, model):
    """
    :param content: sentence
    :param model: model object
    """
    content = re.sub("\d+", "NUM", content)
    content = re.sub("[^\w]", ' ', content)
    content = ' '.join(jieba.cut(openCC.convert(content).replace("\t"," ").replace("\n"," ")))
    return model.predict(content)

if __name__ == '__main__':

    # #load训练好的模型
    # """
    # load_model(path)
    #     Load a model given a filepath and return a model object.
    # """
    classifier = fastText.load_model('reviews_fasttext.bin')

    print(predict('好 差劲', classifier))  
    # <!> todo: 例如 好差劲 这个词, 应当是绝对负面词, 这边需要考虑依存关系(如一般否定词那样处理)
    # <!> todo: 分词部分也有待改进(训练以前的工作)
    # ([['__label__neg'], ['__label__neg']], array([[1.00000989],[0.60022247]]))

    # #测试模型
    # # result = classifier.test(ftest)
    # # print(result)
    # print(classifier.predict(['糟糕', '好 差劲']))
    # print(classifier.predict(['优秀', '很 好']))
    # """
    # ([['__label__neg'], ['__label__neg']], array([[1.00000989],
    #     [0.60022247]]))
    # ([['__label__pos'], ['__label__pos']], array([[1.00000381],
    #     [0.99439901]]))
    # """