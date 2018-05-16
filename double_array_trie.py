# -*- coding:utf-8 -*-

# base
# https://linux.thai.net/~thep/datrie/datrie.html
# http://jorbe.sinaapp.com/2014/05/11/datrie/
# http://www.hankcs.com/program/java/%E5%8F%8C%E6%95%B0%E7%BB%84trie%E6%A0%91doublearraytriejava%E5%AE%9E%E7%8E%B0.html 
# (komiya-atsushi/darts-java | 先建立Trie树，再构造DAT，为siblings先找到合适的空间)
# https://blog.csdn.net/kissmile/article/details/47417277
# http://nark.cc/p/?p=1480

# 不需要构造真正的Trie树，直接用字符串，构造对应node，因为words是排过序的

# todo : error info
# todo : performance test
# todo : resize
# warning: code=0表示叶子节点可能会有隐患(正常词汇的情况下是ok的)
# 修正: 由于想要回溯字符串的效果，叶子节点和base不能重合(这样叶子节点可以继续记录其他值比如频率)，叶子节点code: 0->-1
# 但是如此的话，叶子节点可能会与正常节点冲突？ 找begin的使用应该是考虑到的？

class DATrie(object):

    class Node(object):

        def __init__(self, code, depth, left, right):
            self.code = code
            self.depth = depth
            self.left = left
            self.right = right

    def __init__(self):
        self.MAX_SIZE = 2097152  # 65536 * 32
        self.base = [0] * self.MAX_SIZE
        self.check = [-1] * self.MAX_SIZE  # -1 表示空
        self.used = [False] * self.MAX_SIZE
        self.nextCheckPos = 0  # 详细 见后面->
        self.size = 0  # 记录总共用到的空间

    # 需要改变size的时候调用，这里只能用于build之前。cuz没有打算复制数据.
    def resize(self, size):
        self.MAX_SIZE = size
        self.base = [0] * self.MAX_SIZE
        self.check = [-1] * self.MAX_SIZE
        self.used = [False] * self.MAX_SIZE

    # 先决条件是self.words ordered 且没有重复
    # siblings至少会有一个
    def fetch(self, parent):
        depth = parent.depth

        siblings = []  # size == parent.right-parent.left
        i = parent.left
        while i < parent.right:
            s = self.words[i][depth:]  # '.*'
            if s == '':
                siblings.append(
                    self.Node(code=-1, depth=depth+1, left=i, right=i+1)) # 叶子节点
            else:
                c = ord(s[0])
                if siblings == [] or siblings[-1].code != c:
                    siblings.append(
                        self.Node(code=c, depth=depth+1, left=i, right=i+1)) # 新建节点
                else:  # siblings[-1].code == c
                    siblings[-1].right += 1
            i += 1
        # siblings
        return siblings


    # 在insert之前，认为可以先排序词汇，对base的分配检查应该是有利的
    # 先构建树，再构建DAT，再销毁树
    def build(self, words):
        words = sorted(list(set(words)))  # 去重排序
        self.words = words
        # todo: 销毁_root
        _root = self.Node(code=0, depth=0, left=0, right=len(self.words))
        self.base[0] = 1
        siblings = self.fetch(_root)
        self.insert(siblings, 0)
        # while False:  # 利用队列来实现非递归构造
            # pass
        del self.words
        print("DATrie builded.")


    def insert(self, siblings, parent_base_idx):
        """ parent_base_idx为父节点base index, siblings为其子节点们 """
        # 暂时按komiya-atsushi/darts-java的方案
        # 总的来讲是从0开始分配beigin]
        self.used[parent_base_idx] = True

        begin = 0
        pos = max(siblings[0].code + 1, self.nextCheckPos) - 1
        nonzero_num = 0  # 非零统计
        first = 0  

        begin_ok_flag = False  # 找合适的begin
        while not begin_ok_flag:
            pos += 1
            if pos >= self.MAX_SIZE:
                raise Exception("no room, may be resize it.")
            if self.check[pos] != -1 or self.used[pos]:   # 注意一下check和used的关系?
                nonzero_num += 1  # 已被使用
                continue
            elif first == 0:
                self.nextCheckPos = pos  # 第一个可以使用的位置，记录？
                first = 1

            begin = pos - siblings[0].code  # 对应的begin

            if begin + siblings[-1].code >= self.MAX_SIZE:
                raise Exception("no room, may be resize it.")

            if self.used[begin]:
                continue

            if len(siblings) == 1:
                begin_ok_flag = True
                break

            for sibling in siblings[1:]:
                if self.check[begin + sibling.code] == -1 and self.used[begin + sibling.code] is False:
                    begin_ok_flag = True
                else:
                    begin_ok_flag = False
                    break

        # 得到合适的begin

        # -- Simple heuristics --
        # if the percentage of non-empty contents in check between the
        # index 'next_check_pos' and 'check' is greater than some constant value
        # (e.g. 0.9), new 'next_check_pos' index is written by 'check'.
        
        # todo: 这个写法还需考证
        if (nonzero_num / (pos - self.nextCheckPos + 1)) >= 0.95:
            self.nextCheckPos = pos

        self.used[begin] = True

        # base[begin] 记录 parent chr  -- 这样就可以从节点回溯得到字符串 
        # 想要可以回溯的话，就不能在字符串末尾节点记录值了，或者给叶子节点找个0以外的值？ 0->-1
        self.base[begin] = parent_base_idx


        if self.size < begin + siblings[-1].code + 1:
            self.size = begin + siblings[-1].code + 1
        
        for sibling in siblings:
            self.check[begin + sibling.code] = begin

        for sibling in siblings:  # 由于是递归的情况，需要先处理完check
            # darts-java 还考虑到叶子节点有值的情况，暂时不考虑(需要记录的话，记录在叶子节点上)
            if sibling.code == -1:
                self.base[begin + sibling.code] = -1 * sibling.left - 1
            else:
                new_sibings = self.fetch(sibling)
                h = self.insert(new_sibings, begin + sibling.code)
                self.base[begin + sibling.code] = h

        return begin


    def search(self, word):
        """ 查找单词是否存在 """
        p = 0  # root
        if word == '':
            return False
        for c in word:
            c = ord(c)
            next = abs(self.base[p]) + c
            # print(c, next, self.base[next], self.check[next])
            if next > self.MAX_SIZE:  # 一定不存在
                return False
            # print(self.base[self.base[p]])
            if self.check[next] != abs(self.base[p]):
                return False
            p = next
        
        # print('*'*10+'\n', 0, p, self.base[self.base[p]], self.check[self.base[p]])
        # 由于code=0,实际上是base[leaf_node->base+leaf_node.code]，这个负的值本身没什么用
        # 修正：left code = -1
        if self.base[self.base[p] - 1] < 0 and self.base[p] == self.check[self.base[p] - 1] :  
            return True
        else:  # 不是词尾
            return False


    def common_prefix_search(self, content):
        """ 公共前缀匹配 """
        # 用了 darts-java 写法，再仔细看一下
        result = []
        b = self.base[0]  # 从root开始
        p = 0
        n = 0
        tmp_str = ""
        for c in content:
            c = ord(c)
            p = b
            n = self.base[p - 1]      # for iden leaf

            if b == self.check[p - 1] and n < 0:
                result.append(tmp_str)

            tmp_str += chr(c)
            p = b + c   # cur node
            
            if b == self.check[p]:
                b = self.base[p]  # next base
            else:                 # no next node
                return result

        # 判断最后一个node
        p = b
        n = self.base[p - 1]

        if b == self.check[p - 1] and n < 0:
            result.append(tmp_str)

        return result

    
    def get_string(self, chr_id):
        """ 从某个节点返回整个字符串, todo:改为私有 """
        if self.check[chr_id] == -1:
            raise Exception("不存在该字符。")
        child = chr_id
        s = []
        while 0 != child:
            base = self.check[child]
            print(base, child)
            label = chr(child - base)
            s.append(label)
            print(label)
            child = self.base[base]
        return "".join(s[::-1])


    def get_use_rate(self):
        """ 空间使用率 """
        return self.size / self.MAX_SIZE

if __name__ == '__main__':
    words = ["一举",
            "一举一动",
            "一举成名",
            "一举成名天下知",
            "万能",
            "万能胶"]

    datrie = DATrie()
    datrie.build(words)
    print('-'*10)
    print(datrie.search("一举一动"))
    print('-'*10)
    print(datrie.search("万能胶"))
    print('-'*10)
    print(datrie.common_prefix_search("万能钥匙"))
    print(datrie.common_prefix_search("一举成名天下知"))

    print('-'*10)
    print(datrie.get_string(21520))

    print('-'*10)
    print()