# -*- coding:utf-8 -*-

# basic
# https://github.com/fxsjy/jieba/blob/master/jieba/finalseg/__init__.py

import pickle

MIN_FLOAT = -3.14e100

PROB_START_P = "prob_start.pkl"
PROB_TRANS_P = "prob_trans.pkl"
PROB_EMIT_P = "prob_emit.pkl"

# todo: 要用的话,1 封装; 2 需大一点的语料训练

# 对viterbi, 稍晚简化了一些, 考虑到了obs不存在的情况(当然只是简单的添加极小值)
PrevStatus = {
    'B': 'ES',
    'M': 'MB',
    'S': 'SE',
    'E': 'BM'
}

def load_params():
    start_f = open(PROB_START_P, 'rb')
    start_p = pickle.load(start_f)
    start_f.close()
    trans_f = open(PROB_TRANS_P, 'rb')
    trans_p = pickle.load(trans_f)
    trans_f.close()
    emit_f = open(PROB_EMIT_P, 'rb')
    emit_p = pickle.load(emit_f)
    emit_f.close()
    return start_p, trans_p, emit_p

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # tabular
    path = {}
    for y in states:  # init
        # 如果不存在obs in emit_p, 返回min_float, 表示可能性很低
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        path[y] = [y]
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            em_p = emit_p[y].get(obs[t], MIN_FLOAT)
            (prob, state) = max(
                [(V[t - 1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p, y0) for y0 in PrevStatus[y]])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath

    (prob, state) = max((V[len(obs) - 1][y], y) for y in 'ES')

    return (prob, path[state])

def _cut(sentence):
    pass

def cut(sentence):
    pass

if __name__ == '__main__':

    # 对 Bakeoff 2005 测试集做测试
    test_fn = "icwb2-data/testing/msr_test.utf8"
    output_fn = "icwb2-data/testing/msr_test_seg.utf8"
    start_p, trans_p, emit_p = load_params()
    fw = open(output_fn, 'w')
    with open(test_fn, 'r') as f:
        for line in f:
            obs = line.strip()
            prob, states = viterbi(obs, ('B','M','E','S'), start_p, trans_p, emit_p)
            seg = ""
            for i, tag in enumerate(states):
                if tag == 'S':
                    seg += obs[i]
                    seg += " "
                elif tag == 'B' or tag == 'M':
                    seg += obs[i]
                elif tag == 'E':
                    seg += obs[i]
                    seg += " "
            fw.write(seg+"\n")
    fw.close()
    """ 使用msr语料训练的结果(基于hmm, viteribi)
    === SUMMARY:
    === TOTAL INSERTIONS:   7859
    === TOTAL DELETIONS:    5091
    === TOTAL SUBSTITUTIONS:        16278
    === TOTAL NCHANGE:      29228
    === TOTAL TRUE WORD COUNT:      106873
    === TOTAL TEST WORD COUNT:      109641
    === TOTAL TRUE WORDS RECALL:    0.800
    === TOTAL TEST WORDS PRECISION: 0.780
    === F MEASURE:  0.790
    === OOV Rate:   0.026
    === OOV Recall Rate:    0.363
    === IV Recall Rate:     0.812
    """