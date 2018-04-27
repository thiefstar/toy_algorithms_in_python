# -*- coding:utf-8 -*-

# basic
# flickering.cn/nlp/2014/06/日文分词器-mecab-文档/  + 品詞体系
# http://taku910.github.io/mecab/
# https://www.taodocs.com/p-85211904.html 
#
# http://ictclas.nlpir.org/nlpir/html/readme.htm (ictclas计算所汉语词性标记集)
# https://blog.csdn.net/leiting_imecas/article/details/68485254 (词性)
# requirements.txt
# mecab-python3==0.7
# pynlpir

"""
词性(jieba)
r-代词
v-动词
n-名词
x-字符串
m-数词
f-方位词
q-量词
a-形容词
y-语气词

(nlpir)
代词
名词
动词
形容词
助词
副词
数词
量词
方位词
语气词
标点符号
处所词

(mecab)
名詞
名詞-代名詞 (特别)
名詞-数 (特别)
動詞
記号
"""

import MeCab
import jieba.posseg as pseg
import pynlpir
import re

# 基于长度
# 基于词典(特殊词表?)
# 基于MT? bleu,相似度.. (小牛本地，只有中英)
# 基于词性统计  (暂时对此方法正确性没把握，先不做，还需要增加这方面的知识)

mecab = MeCab.Tagger ("-Ochasen")

def my_jp_split_old(string):
    """ (暂时不用这种，效果还是不如新的好)
    分句, 想保留分隔符，但re.split出现 ValueError: split() requires a non-empty pattern match.
    「」引用中间不分割  (还有一个最后为」的情况，最后判断一下)
    "『うちの子に何をするのよ。』と大声をあげた。・・・」"这种暂时还分不好, 考虑先行删除'・'？, 小牛没有这符号的转换
    """
    MAX_LOC = 10000
    MIN_LOC = -1

    # "\W」\b" - 不分的特征  => 改进
    # pattern = re.compile('(?<=[。？])(?![」』].?「?\b)')  # 区别是?
    pattern = re.compile(r'(?<=[。？])(?![」』]\s?.?「?\b)', re.UNICODE)
    result = []
    tmp = ''
    while string != '':

        s = re.search('「', string)
        quo1_loc = s.span()[0] if s else MAX_LOC  
        s = re.search('」', string)
        quo2_loc = s.span()[0] if s else MIN_LOC  


        s = re.search(pattern, string)
        if s is None:
            tmp += string
            result.append(tmp)
            break
        
        loc = s.span()[0]
        tmp += string[:loc]
        string = string[loc:]

        if loc <= quo1_loc and loc >= quo2_loc and quo1_loc == MAX_LOC: 
            result.append(tmp)
            tmp = ''
        elif loc <= quo1_loc and loc < quo2_loc:
            result.append(tmp)
            tmp = ''
        elif loc < quo2_loc:
            continue
        elif loc > quo1_loc:  # 同时 > quo2_loc
            result.append(tmp)
            tmp = ''

    if result:
        if result[-1] != '' and result[-1].strip()[0] == '」':  # 最后没有标点且最后一段存在」(cuz 这样都没做最后的loc判断)
            last_ = result.pop()
            result[-1] += last_

    return result


def my_split(string):
    """
    原来为my_zh_split, 可以与my_jp_split合并起来了 -> my_split 
    直接使用新的方法，将引号内看作整体保存与队列，后面再换回
    省略号暂时不加，与my_jp_split同步  
    # todo 可以考虑说话部分的分句，
    # 例如‘xxx：“xxx。”xx，xxxx。’
    # 还可分。
    """
    SPLIT_SIGN = '%%%%'  # 需要保证字符串内本身没有这个分隔符

    # 替换的符号用: $PACK$
    SIGN = '$PACK$'
    search_pattern = re.compile('\$PACK\$')
    pack_pattern = re.compile('(“.+?”|（.+?）|《.+?》|〈.+?〉|[.+?]|【.+?】|‘.+?’|「.+?」|『.+?』|".+?"|\'.+?\')')
    pack_queue = []
    pack_queue = re.findall(pack_pattern, string)
    string = re.sub(pack_pattern, SIGN, string)

    pattern = re.compile('(?<=[。？！])(?![。？！])')
    result = []
    while string != '':
        s = re.search(pattern, string)
        if s is None:
            result.append(string)
            break
        loc = s.span()[0]
        result.append(string[:loc])
        string = string[loc:]
    
    result_string = SPLIT_SIGN.join(result)
    while pack_queue:
        pack = pack_queue.pop(0)
        loc = re.search(search_pattern, result_string).span()
        result_string = result_string[:loc[0]] + pack + result_string[loc[1]:]

    return result_string.split(SPLIT_SIGN)


def match(zh_sentences, jp_sentences):
    """
    :param zh_sentences: a list of sentences(zh)
    """
    pass

def split_sentences(content):
    """
    返回[(句子, 长度占比),]
    简单的直接统计长度
    numpy?
    """
    result = []
    count = 0
    sentences = my_split(content)
    for sent in sentences:
        result.append(len(sent))
        count += result[-1]
    return [(sentences[i], numer/count) for i, numer  in enumerate(result)]


def stati_pos(content, lang='zh'):
    """
    :param lang: zh, jp
    统计词性，返回字典，具体词性看相关说明 
    todo: 返回特殊词集合
    """
    pos_count = {}
    if lang == 'zh':
        items = pynlpir.segment(content, pos_english=False)
        for item in items:
            if item[1] is None:
                continue
            pos = item[1]
            pos_count[pos] = pos_count.get(pos, 0) + 1
        
    elif lang == 'jp':
        res = mecab.parse(content)
        for item in res.split('\n'):
            if item == 'EOS':
                break
            pos = item.split('\t')[3]
            pos_count[pos] = pos_count.get(pos, 0) + 1
            
    # 可以扩展en..
    return pos_count




if __name__ =='__main__':

    fn = 'parallel_parag.txt'

    # pynlpir.open()
    # count = 1
    # with open(fn, 'r') as f:
    #     for line in f:
    #         jp_content, zh_content = line.strip().split("###")
    #         print(jp_content)
    #         # print(stati_pos(jp_content, lang='jp'))
    #         print(zh_content)
    #         # print(stati_pos(zh_content))
    #         print()
    #         if count == 20:
    #             break
    #         count += 1

    # pynlpir.close()

    sent = "真是渣渣！！！！！我好无语啊。。。。。都是什么啊，，，，"
    print(my_split(sent))