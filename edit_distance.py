# -*- coding:utf-8 -*-

# import numpy as np

# Levenshtein distance(L氏距离) 三种原子操作 | 默认
INSERT_COST = 1 # 插入
SUBSTITUTE_COST = 1 # 替换  => 如果为2，其实可以去掉这个操作，只要删除插入
DELETE_COST = 1 # 删除

# Damerau,F,J distance（D氏距离）四种原子操作(包括相邻交替)
# 或者叫做 Damerau–Levenshtein distance (wiki)


# 使用动态规矩  # 参考 http://www.doc88.com/p-9532775574400.html
def lev(source_str, target_str):
    source_str_len, target_str_len = len(source_str), len(target_str)

    if source_str_len == 0:
        return target_str_len
    elif target_str_len == 0:
        return source_str_len

    # DP Array
    # mat = np.zeros((source_str_len+1, target_str_len+1))
    mat = [[ 0 for i in range(target_str_len + 1)]
               for j in range(source_str_len + 1)]

    for i in range(1, source_str_len+1):
        mat[i][0] = i
    for j in range(1, target_str_len+1):
        mat[0][j] = j
    for i in range(1, source_str_len+1):
        for j in range(1, target_str_len+1):
            sub_cost = 0 if source_str[i-1] == target_str[j-1] else SUBSTITUTE_COST
            mat[i][j] = min(mat[i-1][j]+INSERT_COST,
                            mat[i-1][j-1]+sub_cost,
                            mat[i][j-1]+DELETE_COST)
    # print(mat)
    return mat[-1][-1]


# Damerau-Levenshtein Distance
# 参考 https://github.com/pupuhime/Levenshtein/blob/master/levenshtein.py and wiki
def dalev(source_str, target_str):
    # Suppose we have two strings 'ac' and 'cba'
    # This is the initialized matrix:
    #
    #                target_str
    #                    c b a
    #              * * * * * * *
    #              * 5 5 5 5 5 *
    #              * 5 0 1 2 3 *
    # source_str a * 5 1 5 5 5 *
    #            c * 5 2 5 5 5 *
    #              * * * * * * *
    #
    source_str_len, target_str_len = len(source_str), len(target_str)
    if source_str_len == 0:
        return target_str_len
    elif target_str_len == 0:
        return source_str_len

    maxdist = source_str_len + target_str_len
    da = {}  # pointer of the last row where a[i] == b[j]
##
    INIT_POS = 2 # initial position of two str ('some'[0] etc.) in the matrix
    ORIGIN = INIT_POS - 1 # the position of '0' in the matrix
    mat = [[ maxdist for i in range(target_str_len + INIT_POS)]
               for j in range(source_str_len + INIT_POS)]
    # mat = maxdist * np.ones((source_str_len+INIT_POS, target_str_len+INIT_POS))
    for i in range(ORIGIN, source_str_len + INIT_POS):
        mat[i][1] = i - ORIGIN
    for j in range(ORIGIN, target_str_len + INIT_POS):
        mat[1][j] = j - ORIGIN
    for i in range(INIT_POS, source_str_len + INIT_POS):
        db = ORIGIN   # pointer of the last column where b[j] == a[i]
        for j in range(INIT_POS, target_str_len + INIT_POS):
            k1 = da.get(source_str[j-INIT_POS], ORIGIN)
            k2 = db
            if source_str[i - INIT_POS] == target_str[j - INIT_POS]:
                cost = 0
                db = j
            else:
                cost = SUBSTITUTE_COST
            mat[i][j] = min(mat[i-1][j] + DELETE_COST,
                            mat[i][j-1] + INSERT_COST,
                            mat[i-1][j-1] + cost,
                            mat[k2-1][k1-1] + (i-k1-1) + 1 + (j-k2-1)) # 这个1为 tran_cost ?
            da[source_str[i-INIT_POS]] = i  # pointer of row? (只记录最近的)
    return mat[-1][-1]

# and ..
# Optimal string alignment distance (osa)e

if __name__ == "__main__":
    str1 = "stop"
    str2 = "sotp"

    print(lev(str1, str2))
    print(dalev(str1, str2))