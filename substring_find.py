# -*- coding:utf-8 -*-

import prime

# kmp with next array
class KMP(object):

    def _calc_next_array(self, pat, length):
        next = [0]
        cur_next = 0
        for i in range(1, length):
            while cur_next != 0 and pat[i] != pat[cur_next]:
                cur_next = next[cur_next-1]
            if pat[i] == pat[cur_next]:
                cur_next += 1
            next.append(cur_next)
        return next

    def search(self, pat, txt):
        M = len(pat)
        N = len(txt)
        next = self._calc_next_array(pat, M)
        j = 0
        for i in range(N):
            while j!=0 and txt[i]!=pat[j]:
                j = next[j-1]
            if txt[i] == pat[j]:
                j += 1
        if j == M:
            return N - M  # 找到匹配开始处index
        else:
            return N  # 为找到匹配, 返回尾部index

# kmp with dfa
class KMP_DFA(object):

    def search(self, pat, txt):
        M = len(pat)
        N = len(txt)
        chars = {}
        R = 1
        for p in set(pat):
            chars[p] = R
            R += 1
        dfa = [[0 for _ in range(M)] for i in range(R)]
        dfa[chars.get(pat[0], 0)][0] = 1
        X = 0
        for j in range(1, M):  # 计算dfa
            for c in range(R):
                dfa[c][j] = dfa[c][X]
            dfa[chars.get(pat[j], 0)][j] = j+1
            X = dfa[chars.get(pat[j], 0)][X]
        j = 0
        for i in range(N):
            if j == M or i == N:
                break
            j = dfa[chars.get(txt[i], 0)][j]
            if j == M:
                return i - M + 1  # 找到匹配开始处index
        return N  # 未找到匹配, 返回尾部index

# bm
class Boyer_Moore(object):

    def search(self, pat, txt):
        M = len(pat)
        N = len(txt)
        chars = list(set(pat+txt))
        R = len(chars) #可以处理所以字符
        right = dict().fromkeys(chars, -1)
        for j in range(M):
            right[pat[j]] = j  # 最右位置
        i = 0
        while i<=N-M:
            skip = 0
            for j in range(M)[::-1]:
                if pat[j] != txt[i+j]:
                    skip = j - right[txt[i+j]]
                    if skip < 1:
                        skip = 1
                    break
            if skip == 0:
                return i
            i += skip
        return N  # 未找到匹配, 返回尾部index

class Rabin_Karp(object):

    """
    指纹字符串查找
    """
    def __init__(self, txt):
        self.txt = txt
        self.Q = prime.RandomPrime.generate(20)  # 随机键的hash值与pat的hash值冲突的概率将小于10^-20
        self.chars = {}
        self.char_set = set(txt)
        for index, c in enumerate(self.char_set, 1):  # 0将用于表示不存在的字符
            # print(index, c)
            self.chars[c] = index
        self.R = len(self.char_set) + 1  # 字符表大小


    def hash(self, key, M, Q, R=10):
        """
        除余法  hash
        :param key:
        :param M: key[:M]
        :param Q: % Q
        :param R: 进制
        :return:
        """
        h = 0
        for j in range(M):
            h = (self.R * h + self.chars.get(key[j], 0)) % Q
        return h

    def check(self, pat, txt, i): # 蒙特卡洛算法验证? 或者直接对比验证
        return True

    def search(self, pat):
        M = len(pat)
        N = len(self.txt)
        RM = 1  # R^(M-1) % Q
        for _ in range(M-1):
            RM = (self.R * RM) % self.Q

        pat_hash = self.hash(pat, M, self.Q, self.R)

        txt_hash = self.hash(self.txt, M, self.Q, self.R)
        if pat_hash == txt_hash and self.check(pat, self.txt, 0):
            return 0
        for i in range(M, N):
            txt_hash = (txt_hash + self.Q - RM*self.chars.get(self.txt[i-M], 0) % self.Q) % self.Q
            txt_hash = (txt_hash*self.R + self.chars.get(self.txt[i], 0)) % self.Q
            if pat_hash == txt_hash:
                if self.check(pat, self.txt, i-M+1):
                    return i-M+1
        return N  # 失败返回 N 长度