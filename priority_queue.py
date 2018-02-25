# -*- coding:utf-8 -*-

class MaxPQ(object):
    """
    优先队列
    using swim and sink function. And pq is compose of number
    """
    N = 0  # size of Priroty Queue
    def __init__(self, maxN):
        self.pq = [None for i in range(maxN+1)]

    def is_empty(self):
        return self.N == 0

    def size(self):
        return self.N

    def insert(self, v):
        self.N += 1
        self.pq[self.N] = v
        self.swim(self.N)
        return True

    def del_max(self):
        max = self.pq[1]
        self.exch(1, self.N)
        self.pq[self.N] = None
        self.N -= 1
        self.sink(1)
        return max


    def less(self, i, j):
        return self.pq[i] < self.pq[j]

    def exch(self, i, j):
        self.pq[i], self.pq[j] = self.pq[j], self.pq[i]
        return True

    def swim(self, k):
        while k>1 and self.less(k//2, k):
            self.exch(k//2, k)
            k //= 2
        return True

    def sink(self, k):
        while 2*k <= self.N:
            j = 2*k
            if j<self.N and self.less(j, j+1):
                j += 1
            if not self.less(k, j):
                break
            self.exch(k, j)
            k = j
        return True


class IndexMinPQ(object):
    """
    索引优先队列
    """

    def __init__(self, maxN):
        self.N = 0  #  size of Indexed Priroty Queue
        self.maxN = maxN
        self.keys = [None for i in range(maxN + 1)]  # 索引与值的对应， [0, maxN]
        self.pq = [None for i in range(maxN + 1)]  # 索引二叉堆， [1, maxN]
        self.qp = [-1 for i in range(maxN + 1)]  # 逆序，满足qp[pq[i]] = pq[qp[i]] = i


    def is_empty(self):
        return self.N == 0

    def size(self):
        return self.N

    def _greater(self, i, j):
        return self.keys[self.pq[i]] > self.keys[self.pq[j]]

    def _exch(self, i, j):
        self.pq[i], self.pq[j] = self.pq[j], self.pq[i]  # 只交换索引而不必交换元素
        self.qp[self.pq[i]] = i  # 还要更新qp
        self.qp[self.pq[j]] = j
        return True

    def _swim(self, k):
        while k>1 and self._greater(k//2, k):
            self._exch(k//2, k)
            k //= 2
        return True

    def _sink(self, k):
        while 2*k <= self.N:
            j = 2*k
            if j<self.N and self._greater(j, j+1):
                j += 1
            if not self._greater(k, j):
                break
            self._exch(k, j)
            k = j
        return True

    def insert(self, k, item):
        """
        插入索引为k的元素item
        :param k:
        :param item:
        :return:
        """
        if not self.contains(k):
            self.N += 1
            self.pq[self.N] = k
            self.qp[k] = self.N
            self.keys[k] = item
            self._swim(self.N)
        return True

    def change(self, k, item):
        self.keys[k] = item
        # 由于和k关联的新值可能大于原来的值（此时需要下沉），
        # 也有可能小于原来的值（此时需要上浮），为了简化代码，既上浮又下沉，就囊括了这两种可能。
        self._swim(self.qp[k])
        self._sink(self.qp[k])
        return True

    def contains(self, k):
        """
        是否存在索引为k的元素
        :param k:
        :return:
        """
        return self.qp[k] != -1

    def delete(self, k):
        """
        删除索引k & 其相关联的元素
        :param k:
        :return:
        """
        if not self.contains(k):
            raise Exception("不存在索引为%s的元素！" % k)
        loc = self.qp[k]
        self._exch(loc, self.N)
        self.N -= 1
        self._swim(loc)
        self._sink(loc)
        self.keys[k] = None
        self.qp[k] = -1

        return True

    def min(self):
        """
        返回最小元素
        :return:
        """
        return self.keys[self.pq[1]]

    def min_index(self):
        """
        返回最小元素的索引
        :return:
        """
        return self.pq[1]

    def del_min(self):
        """
        删除最小元素，返回对应索引
        :return:
        """
        if self.is_empty():
            raise Exception("队列为空，不能执行del_min操作！")
        index_of_min = self.pq[1]
        self._exch(1, self.N)
        self.N -= 1
        self._sink(1)
        self.keys[index_of_min] = None
        self.qp[index_of_min] = -1

        return index_of_min
