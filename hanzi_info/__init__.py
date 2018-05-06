# -*- coding:utf-8 -*-

# basic
# cnblogs.com/Comero/p/8995261.html
# todo
# 兼容python2

# -*- coding:utf-8 -*-

# basic  (py3)
# cnblogs.com/Comero/p/8995261.html
# todo
# 兼容python2

kangxi_radical_fn = "kangxi_radical.pkl"
radicals_fn = "radicals.pkl"
strokes_fn = "strokes.txt"

import pickle

def load_data():

    # [radical,...] idx = kangxi_id - 1
    with open(kangxi_radical_fn, 'rb') as fr:
        kangxi_radical = pickle.load(fr)  

    # {unicode: "kangxi_radical_id.additional_strokes"}
    with open(radicals_fn, 'rb') as fr:
        radicals = pickle.load(fr) 

    # 13312-64045 and 131072-194998
    # 映射到大小为114661的数组中，值为笔画数
    strokes = []
    with open(strokes_fn, 'r') as fr:
        for line in fr:
            strokes.append(int(line.strip()))

    return kangxi_radical, radicals, strokes


kangxi_radical_table, radicals_table, strokes_table = load_data()
kangxi_radical_table = [""] + kangxi_radical_table


def get_radical(c):

    assert type(c)==str, "TypeError, need str type (a char) for funcation get_radical()"
    unicode_ = ord(c)
    items = radicals_table.get(unicode_, '0.-1').split('.')
    return (kangxi_radical_table[int(items[0])], int(items[1]))


def get_stroke(c):
    # 如果返回 0, 则也是在unicode中不存在kTotalStrokes字段

    assert type(c)==str, "TypeError, need str type (a char) for funcation get_stroke()"

    unicode_ = ord(c)

    if 13312 <= unicode_ <= 64045:
        return strokes_table[unicode_-13312]
    elif 131072 <= unicode_ <= 194998:
        return strokes_table[unicode_-80338]
    else:
        print("c should be a CJK char, or not have stroke field in unihan data.")
        return 0
