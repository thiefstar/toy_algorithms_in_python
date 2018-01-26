# -*- coding:utf-8 -*-

import random

class PrimeTest(object):
    # http://www.matrix67.com/blog/archives/234
    a_candidate = [(2,),
                   (2, 7, 61),
                   (2, 3, 5, 7, 11, 13, 17),
                   (2, 3, 7, 61, 24251)]

    def _pow_mod(self, a, d, n):
        """

        :param a:
        :param d:
        :param n:
        :return:  a**d % n
        """
        if d==0:
            return 1
        elif d==1:
            return a
        elif d&1==0:
            return self._pow_mod(a*a%n, d//2, n) % n
        else:
            return self._pow_mod(a*a%n, d//2, n) * a % n

    def Miller_Rabin(self, n, a):
        """
        :param a:
        :param p: number to be tested
        :return:
        """
        if n == 2:
            return True
        if n<=1 or n&1==0:
            return False
        d = n - 1
        while d&1==0:
            d = d>>1
        t = self._pow_mod(a, d, n)
        while d!=n-1 and t!=1 and t!=n-1:  #补上一开始先除的部分
            t = (t*t) % n
            d <<=1
            # print(d, t)
        return ((t==n-1) or (d&1==1))

    def is_prime(self, n, level=0):
        """

        :param n:
        :param level: 严格程度 0-3, 3最高
        :return:
        """
        for a in self.a_candidate[0]:
            if not self.Miller_Rabin(n, a):
                return False
        return True

class RandomPrime(object):
    """"""
    @staticmethod
    def generate(n, level=0):
        """

        :param n: 长度
        :param level: 严格程度 0-3, 3最高
        :return:
        """
        prime_test = PrimeTest()
        num = random.randint(10**(n-1), 10**n-1)
        while not prime_test.is_prime(num):
            num = random.randint(10**(n-1), 10**n-1)
        return num