# -*- coding:utf-8 -*-

# basic
# 统计学习方法
# http://www.hankcs.com/ml/naive-bayesian-method.html 
# https://github.com/isnowfy/snownlp
# https://blog.csdn.net/stdcoutzyx/article/details/9285001
# (基本保持一致吧, snownlp中基于NB的情感判别, 主要是记录)

from math import log, exp
import pickle

class LaplaceSmoothingCount(object):
    """
    统计时使用拉布拉斯平滑(加一平滑), 为了解决零概率问题.
    """
    def __init__(self):
        self.d = {}  # [词-词频]的map
        self.total = 0.0  # 全部词的词频
        self.none = 1  # 当一个词不存在的时候，它的词频（等于0+1）
 
    def exists(self, key):
        return key in self.d
 
    def get_sum(self):
        return self.total
 
    def get(self, key):
        if not self.exists(key):
            return False, self.none
        return True, self.d[key]
 
    def get_prob(self, key):
        """
        估计先验概率
        :param key: 词
        :return: 概率
        """
        return float(self.get(key)[1]) / self.total
 
    def samples(self):
        """
        获取全部样本
        :return:
        """
        return self.d.keys()
 
    def add(self, key, value):
        self.total += value
        if not self.exists(key):  # 平滑
            self.d[key] = 1
            self.total += 1
        self.d[key] += value


class Bayes(object):
    def __init__(self):
        self.d = {}  # {标签:概率,}
        self.total = 0  # 全部词频
 
 
    def train(self, data):
        for d in data:  # d是[[词链表], 标签]
            c = d[1]  # c是分类(标签)
            if c not in self.d:
                self.d[c] = LaplaceSmoothingCount()  # d[c]是概率统计工具
            for word in d[0]:
                self.d[c].add(word, 1)  # 统计词频
        self.total = sum(map(lambda x: self.d[x].get_sum(), self.d.keys()))
 
    def classify(self, x):
        tmp = {}
        for c in self.d:  # 分类
            tmp[c] = log(self.d[c].get_sum()) - log(self.total)  # P(Y=ck)
            for word in x:
                tmp[c] += log(self.d[c].get_prob(word))          # P(Xj=xj | Y=ck)
        ret, prob = 0, 0
        for c in self.d:
            now = 0
            try:
                for otherc in self.d:
                    now += exp(tmp[otherc] - tmp[c])            # 将对数还原为1/p
                now = 1 / now
            except OverflowError:
                now = 0
            if now > prob:
                ret, prob = c, now
        return (ret, prob)


if __name__ == '__main__':

    class Sentiment(object):
        def __init__(self):
            self.classifier = Bayes()
    
        def segment(self, sent):
            words = sent.split(' ')
            return words
    
        def train(self, neg_docs, pos_docs):
            data = []
            for sent in neg_docs:
                data.append([self.segment(sent), u'neg'])
            for sent in pos_docs:
                data.append([self.segment(sent), u'pos'])
            self.classifier.train(data)
    
        def classify(self, sent):
    
            return self.classifier.classify(self.segment(sent))

    s = Sentiment()
    s.train(['糟糕', '好 差劲'], ['优秀', '很 好']) # 空格分词
    
    print(s.classify("好 优秀"))
    print(s.classify("好 差劲"))