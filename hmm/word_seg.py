# -*- coding:utf-8 -*-

# basic
# https://github.com/fxsjy/jieba/blob/master/jieba/finalseg/__init__.py

import pickle

MIN_FLOAT = -3.14e100

PROB_START_P = "prob_start.plk"
PROB_TRANS_P = "prob_trans.plk"
PROB_EMIT_P = "prob_emit.plk"

# 稍晚简化了一些
PrevStatus = {
    'B': 'ES',
    'M': 'MB',
    'S': 'SE',
    'E': 'BM'
}

def load_params():
    pass

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # tabular
    path = {}
    for y in states:  # init
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

def _cat(sentence):
    pass

def cat(sentence):
    pass