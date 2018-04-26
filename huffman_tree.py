# -*- coding:utf-8 -*-

# basic
# 算法(第4版)
# https://github.com/shaharrose/HuffmanTree/blob/master/HuffmanTree.py
# https://github.com/shaharrose/HuffmanTree/blob/master/Parser.py
# docs
# docs.python.org/3/library/functools.html#functools.total_ordering
# requirements.txt
# progressbar2
# bitstring

# <!> 一开始构想的是想支持中文字符等，但由于python 按位的二进制文件操作比较难..
# (有中文的话, header只能用unicode, 因为utf-8不等长; )
# header直接用字符串代替, 存的时候使用unicode(读取的时候转成字符串就无所谓编码了), 仅作为学习

from functools import total_ordering
import heapq
import progressbar
from bitstring import BitArray
import os

@total_ordering
class Node(object):

    def __init__(self, val, freq, left=None, right=None):
        """

        """
        self.val = val
        self.freq = freq
        self.left = left
        self.right = right
        
    def is_leaf(self):
        return self.left is None and self.right is None

    @staticmethod
    def encode_node(node):
        if node.is_leaf():
            return [1, node.val]
        else:
            return [0] + Node.encode_node(node.left) + Node.encode_node(node.right)

    def __eq__(self, other):
        return self.freq == other.freq

    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanTree(object):

    def __init__(self, freq_dict):
        """
        :param freq_dict: {c:freq, ...}
        """
        self._root = None
        self.st = {}
        self.freq_dict = freq_dict

    def get_root(self):
        return self._root

    def copy(self, root):
        """
        get a huffmantree object by the tree root node.
        no need to build_trie()
        """
        self._root = root

    def build_tire(self):
        """
        构造huffman tree
        """
        h = []
        # 初始化优先队列
        for c, freq in self.freq_dict.items():
            heapq.heappush(h, Node(c, freq))

        for _ in range(len(h) - 1):
            # 合并比较小的两颗树
            x = heapq.heappop(h)
            y = heapq.heappop(h)
            parent = Node('\0', x.freq+y.freq, x, y)
            heapq.heappush(h, parent)
        self._root = heapq.heappop(h)
    
    def build_code(self):
        """
        通过遍历单词查找树来构造编译表
        """
        self._build_code(self._root, '')
        return self.st

    def _build_code(self, x, code):
        """
        :param x: node
        :param code: save the code in recur
        """
        if x.is_leaf():
            self.st[x.val] = code
            return
        self._build_code(x.left, code + '0')
        self._build_code(x.right, code + '1')

    def encode_tree(self):
        return Node.encode_node(self._root)

    def get_value_for_huffman_code(self, binary):
        cur = self._root
        while not cur.is_leaf():
            if not binary:
                break
                # raise ValueError
            if str(binary[0]) == "0":
                cur = cur.left
            elif str(binary[0]) == "1":
                cur = cur.right
            binary = binary[1:]
        return binary, cur.val  # 剩下的binary

    def get_huffman_code_for_value(self, val):
        """
        need build_code before get_huffman_code_for_value
        """
        return self.st.get(val, None)


class Parser(object):

    # -*- compress -*-

    def tree_to_file_header(self, huff):
        """
        :param ht: huffman tree instance
        return a string of header.
        """
        encoded = huff.encode_tree()
        # print(encoded)
        bytess = ''
        for d in encoded:
            if isinstance(d, str):
                bytess += d
            else:
                bytess += str(d)  # 0/1
        # header + spilt_sign
        return bytess + bytes.decode(b'\x03\x03')

    def compress_data(self, tree, data):
        compressed = []
        for c in data:
            compressed.append(tree.get_huffman_code_for_value(c))
        return ''.join(compressed)

    def binary_to_file_data(self, binary):
        mod = len(binary) % 8
        if mod > 0:
            binary = '0' * (8 - mod) + binary
        data = []
        for index in range(len(binary) // 8):
            substring = binary[index * 8: index * 8 + 8]
            byteval = BitArray(bin=substring)
            data.append(byteval.uint)
        return bytearray(data)

    def compress_file(self, uncompressed):
        """
        :param uncompressed: a file to compress.
        """
        print('Compressing')
        bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
        file_handle = open(uncompressed, 'r')
        file_data = file_handle.read()
        file_handle.close()

        huff = HuffmanTree(generate_freq_dict(file_data))
        huff.build_tire()
        huff.build_code()
        header = self.tree_to_file_header(huff)

        binary_data = '1' + self.compress_data(huff, file_data)  # add 1? 这个1有影响，找一下原因(来自文件方面？)
        compressed = self.binary_to_file_data(binary_data)

        file_name = uncompressed + '.huff'
        new_file_data = header.encode('unicode_escape') + compressed
        try:
            os.remove(file_name)
        except:
            pass
        new_handle = open(file_name, 'wb')
        new_handle.write(new_file_data)
        new_handle.close()
        bar.finish()

    # -*- decompress -*-

    def header_to_tree(self, header):
        root = self._read_node(self._header_generator(header))
        ht = HuffmanTree({})
        ht.copy(root)
        return ht

    def _header_generator(self, header):
        for c in header:
            yield c

    def _read_node(self, gen):
        try:
            nex = next(gen)
        except:
            return None
        if int(nex) == 1:
            return Node(next(gen), 0)
        left = self._read_node(gen)
        right = self._read_node(gen)
        n = Node('\0', 0)
        n.left = left
        n.right = right
        return n

    def data_to_bits(self, data):
        final_bits = ""
        for index, b in enumerate(data):
            binary = str(bin(ord(b))[2:])
            if len(binary) < 8 and index > 0:
                binary = '0' * (8 - len(binary)) + binary
            final_bits += binary
        return final_bits

    def decompress_file(self, compressed_file_name):
        print('Decompressing')
        bar = progressbar.ProgressBar(maxval=100, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        file_handle = open(compressed_file_name, 'rb')
        file_data = file_handle.read()
        file_handle.close()

        # h_length 是否需要记录长度信息?
        # only_header = string header
        # compressed_data = file_data[h_length + 1:]
        split_ = str(file_data).split(r'\\x03\\x03')
        only_header = eval(split_[0].strip()+"'").decode('unicode_escape')  # 使用eval()是否需要注意什么，或者可以用其他更安全的方法?
        compressed_data = eval("'" + '\x03\x03'.join(split_[1:]))  # 补全 compressed_data.
        huff = self.header_to_tree(only_header)

        compressed_bits = self.data_to_bits(compressed_data)[1:]
        original_length = float(len(compressed_bits))
        uncompressed_data = ""
        while len(compressed_bits) > 0:
            bar.update(100 * (1 - (len(compressed_bits) / original_length)))
            binary, data = huff.get_value_for_huffman_code(compressed_bits)
            compressed_bits = binary
            uncompressed_data += data

        new_file = 'out/' + '.'.join(compressed_file_name.split('.')[:-1])
        new_handler = open(new_file, 'w')
        new_handler.write(uncompressed_data)
        new_handler.close()

        bar.finish()



def generate_freq_dict(string, from_file=False):
    """
    统计char频率; 如果from_file is True, string is the fn
    todo : 考虑大数据量(来自文件, 迭代的形式)
    """
    freq_dict = {}
    if from_file:
        with open(string, 'r') as f:  # r or rb?
            for line in f:
                for c in line:
                    freq_dict[c] = freq_dict.get(c, 0) + 1
    else:
        for c in string:
            freq_dict[c] = freq_dict.get(c, 0) + 1
    
    return freq_dict

if __name__ == '__main__':
    # s = "AAAAAABCCCCCC大的阿 D 是DEEEEE"

    # ht = HuffmanTree(generate_freq_dict(s))
    # ht.build_tire()
    # print(ht.build_code())
    # print(Node.encode_node(ht.get_root()))
    # print()
    # parser = Parser()
    # header = parser.tree_to_file_header(ht)

    # header = header.strip(bytes.decode(b'\x03\x03'))
    # print(header)
    # new_ht = parser.header_to_tree(header)
    
    # q = [new_ht.get_root()]
    # while q != []:
    #     node = q.pop()
    #     if node.left:
    #         q.append(node.left)
    #     if node.right:
    #         q.append(node.right)
    #     if node.is_leaf():
    #         print(node.val)

    # print(parser.data_to_bits("abc"))

    p = Parser()
    p.compress_file('test_huffman.txt')

    p.decompress_file('test_huffman.txt.huff')