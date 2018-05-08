# -*- coding:utf-8 -*-

# base
# https://github.com/jon-stewart/aho_corasick/blob/master/fsm.py

class AhoCorasickAutomation(object):

    class Node(object):
        def __init__(self, ch):
            self.ch = ch
            self.out = ""
            self.children = {}
            self.fail = None
        
        def goto(self, ch):
            if ch in self.children.keys():
                return self.children[ch]
            else:
                return None

        def add(self, node):
            self.children[node.ch] = node
            return node

    def __init__(self):
        self._root = self.Node("")

    def add(self, pattern):
        """
        add a pattern to tree.
        """
        node = self._root

        for ch in pattern:
            if node.goto(ch):
                node = node.goto(ch)
            else:
                node = node.add(self.Node(ch))
        node.out = pattern

    def build_tree(self, patterns):
        """
        :param patterns: like words
        :type patterns: list
        """
        for pattern in patterns:
            self.add(pattern)

    def buil_fail(self):
        stack = [self._root]
        while len(stack):
            node = stack.pop(0)
            ptr = None
            for ch, next_node in node.children.items():
                # k is ch, and v is next_node
                    ptr = node.fail
                    while ptr and not ptr.goto(ch):
                        ptr = ptr.fail
                    if ptr:
                        next_node.fail = ptr.goto(ch)
                    else:
                        next_node.fail = self._root
                    stack.append(next_node)
        

    def match(self, text):
        node = self._root
        length = len(text)
        for i, c in enumerate(text):
            while node and not node.goto(c):
                node = node.fail
            if not node:
                node = self._root
            else:
                node = node.goto(c)

            if node.out:
                print(i+1-len(node.out), node.out)


if __name__ == '__main__':
    aca = AhoCorasickAutomation()
    aca.build_tree(["a", "ab", "bc", "aab", "aac", "bd"])
    aca.buil_fail()
    aca.match("aabcbd")