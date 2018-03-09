# -*- coding:utf-8 -*-


class TrieST(object):

    class Node(object):
        def __init__(self):
            self.__R = 256  # 基数
            self.__next = [ None for _ in range(self.__R)]
            self.__val = None
    R = 256
    root = Node()

    def get(self, key):
        """
        get the value of key
        :param key:
        :return:
        """
        x = self._get(self.root, key, 0)
        if x is None:
            return None
        return x.val

    def _get(self, node, key, d):
        if node is None:
            return None
        if d == len(key):
            return node
        c = key[d]
        return self._get(node.next[ord(c)], key, d+1)

    def put(self, key, val):
        self.root = self._put(self.root, key, val, 0)

    def _put(self, node, key, val, d):
        if node is None:
            node = self.Node()
        if d == len(key):
            node.val = val
            return node
        c = key[d]
        node.next[ord(c)] = self._put(node.next[ord(c)], key, val, d+1)
        return node

    def size(self):
        return self._size(self.root)

    def _size(self, node):
        if node is None:
            return 0

        cnt = 0
        if node.val != None:
            cnt += 1
        for c in range(self.R):
            cnt += self._size(node.next[c])
        return cnt

    def keys(self):
        return self.keys_with_prefix("")

    def keys_with_prefix(self, pre):
        q = []
        self._collect(self._get(self.root, pre, 0), pre, q)
        return q

    def _collect(self, node, pre, q):
        if node is None:
            return
        if node.val != None:
            q.append(pre)
        for c in range(self.R):
            self._collect(node.next[c], pre + chr(c), q)

    def keys_that_match(self, pat):
        q = []
        self._collect_that_match(self.root, "", pat, q)
        return q

    def _collect_that_match(self, node, pre, pat, q):
        d = len(pre)
        if node is None:
            return
        pat_length = len(pat)
        if d == pat_length and node.val != None:  # 匹配
            q.append(pre)
        if d == pat_length:
            return  # 仅仅只是找到这样一个前缀(没有匹配到键)

        next = pat[d]
        for c in range(self.R):
            if next == '.' or next == chr(c):
                self._collect_that_match(node.next[c], pre+chr(c), pat, q)

    def longest_prefix_of(self, s):
        length = self._search(self.root, s, 0, 0)
        return s[:length]

    def _search(self, node, s, d, length):
        if node is None:
            return length
        if node.val != None:
            length = d  #更新 length
        if d == len(s):
            return length
        c = s[d]
        return self._search(node.next[ord(c)], s, d+1, length)

    def delete(self, key):
        self.root = self._delete(self.root, key, 0)

    def _delete(self, node, key, d):
        if node is None:
            return
        if d == len(key):
            node.val = None
        else:
            c = key[d]
            node.next[ord(c)] = self._delete(node.next[ord(c)], key, d+1)  # 要先将单词结尾置空

        if node.val != None:
            return node  # 指node是某个词结尾，有用，不删
        for c in range(self.R):
            if node.next[c] != None:
                return node  # 指 node 还有其他词组成，不删
        return None


class TST(object):
    """

    """
    class Node(object):
        def __init__(self, c):
            self.c = c
            self.left, self.mid, self.right = None, None, None
            self.val = None

    R = 256  #基数
    root = None

    def get(self, key):
        """
        get the value of key
        :param key:
        :return:
        """
        x = self._get(self.root, key, 0)
        if x is None:
            return None
        return x.val

    def _get(self, x, key, d):
        """

        :param x: root node
        :param key:
        :param d: key[d]
        :return: a node
        """
        if x is None:
            return None
        if len(key) == 0:
            return x
        c = key[d]
        if c < x.c:
            return self._get(x.left, key, d)
        elif c > x.c:
            return self._get(x.right, key, d)
        elif d < len(key) - 1:
            return self._get(x.mid, key, d+1)
        else:
            return x

    def put(self, key, val):
        self.root = self._put(self.root, key, val, 0)

    def _put(self, x, key, val, d):
        c = key[d]
        if x is None:
            x = self.Node(c)
        if c < x.c:
            x.left = self._put(x.left, key, val, d)
        elif c > x.c:
            x.right = self._put(x.right, key, val, d)
        elif d < len(key) - 1:
            x.mid = self._put(x.mid, key, val, d+1)
        else:
            x.val = val
        return x

    def longest_prefix_of(self, s):
        length = self._search(self.root, s, 0, 0)
        return s[:length]

    def _search(self, x, s, d, length):
        if x is None:
            return length
        if x.val != None:
            length = d  # 更新 length
        c = s[d]
        if c < x.c:
            return self._search(x.left, s, d, length)
        elif c > x.c:
            return self._search(x.right, s, d, length)
        elif d < len(s) - 1:
            return self._search(x.mid, s, d+1, length)
        else:
            return length

    def keys(self):
        q = []
        x = self.root
        if x is None:
            return q
        fq = []  # 首字母
        while not x.left is None:
            x = x.left
            fq.append((x, x.c))
        fq = fq[::-1]
        x = self.root        fq.append((x, x.c))
        while not x.right is None:
            x = x.right
            fq.append((x, x.c))
        for x, pre in fq:  # 思路是先获得首字母，然后按 keys_with_prefix来添加单词(x记录，用来节省一部查找节点)
            if x.val != None:
                q.append(pre)
            self._collect(x.mid, pre, q)
        return q

    def keys_with_prefix(self, pre):
        q = []
        x = self._get(self.root, pre, 0) # 从x节点开始的所有单词
        if x.val != None:
            q.append(pre)
        self._collect(x.mid, pre, q)
        return q

    def _collect(self, x, pre, q):
        if x is None:
            return
        if x.val != None:
            q.append(pre+x.c)
        self._collect(x.left, pre, q)
        self._collect(x.mid, pre+x.c, q)
        self._collect(x.right, pre, q)
        return


    def keys_that_match(self, pat):
        if pat == "":
            return []
        q = []
        self._collect_that_match(self.root, "", pat, q)
        return q

    def _collect_that_match(self, x, pre, pat, q):
        """

        :param x: 以x节点开始进行匹配
        :param pre: 记录已经匹配的字符串
        :param pat: 匹配模式，长度至少为1
        :param q: 记录已经匹配的单词
        :return:
        """
        d = len(pre)
        if x is None:
            return
        pat_length = len(pat)
        if d == pat_length:
            return
        cur = pat[d]  # 当前匹配模式字符
        if d == pat_length - 1 and x.val != None:
            if cur == '.' or cur == x.c:
                q.append(pre + x.c)

        if cur == '.':
            self._collect_that_match(x.left, pre, pat, q)
            self._collect_that_match(x.mid, pre+x.c, pat, q)
            self._collect_that_match(x.right, pre, pat, q)
        elif cur < x.c:
            self._collect_that_match(x.left, pre, pat, q)
        elif cur > x.c:
            self._collect_that_match(x.right, pre, pat, q)
        elif cur == x.c:
            self._collect_that_match(x.mid, pre + x.c, pat, q)


    def delete(self):
        # 类似二叉查找树的删除
        pass




if __name__ == "__main__":

    t = TST()
    t.put("apple", 1)
    t.put("abc", 2)
    t.put("agc", 8)
    t.put("abandon", 3)
    t.put("bride", 4)
    t.put("bridegroom", 5)
    t.put("good", 6)
    t.put("b", 7)

    print(t.get("good"))
    print(t.get("ab"))
    print(t.get("abc"))

    print(t.keys())
    print(t.keys_with_prefix("a"))

    print(t.keys_that_match("a.c"))

    # print(t.delete("bride"))
    # print(t.delete("abandon"))

