# -*- coding:utf-8 -*-
# Python implementation of BLEU and smooth-BLEU

# basic
# https://en.wikipedia.org/wiki/BLEU
# https://www.nltk.org/_modules/nltk/translate/bleu_score.html  # nltk implementation
# https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py  # 这两种可以比较一下速度
# https://www.cnblogs.com/by-dream/p/7679284.html  # zh explain

# other
# https://github.com/google/seq2seq/blob/master/seq2seq/metrics/bleu.py # use multi-bleu.perl
# 

import sys
import math
import fractions
import warnings
from collections import Counter


try:
    fractions.Fraction(0, 1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    from nltk.compat import Fraction


def sentence_bleu(references, hypothesis, max_order=4, smoothing_function=None, emulate_multibleu=False, auto_reweigh=False):
    """
    不用weights 和 auto_reweigh，直接按 Wn = 1/n 来
    """
    return corpus_bleu([references], [hypothesis],
                        max_order, smoothing_function,
                        emulate_multibleu)

def corpus_bleu(list_of_references, hypotheses, max_order=4, smoothing_function=None, emulate_multibleu=False, auto_reweigh=False):
    """
    :param references: a corpus of lists of reference sentences, w.r.t. hypotheses
    :type references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param max_order: maximum n of n-grams
    :param emulate_multibleu: round(var, 4) if true.
    :param auto_reweigh: when hyp_len < 4, max_order = hyp_lengths (corpus-level hyp_lengths < 4?)
    """
    p_numerators = Counter() # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter() # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0
    assert len(list_of_references) == len(hypotheses), "The number of hypotheses and their reference(s) should be the same"
    
    for references, hypothesis in zip(list_of_references, hypotheses):
        for i, _ in enumerate(range(1, max_order+1)):  # i -> n
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        # corpus-level hypothesis and reference counts.
        hyp_len =  len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len) 

    # if none macth even uni-gram
    if p_numerators[1] == 0:
        return 0

    if auto_reweigh and hyp_lengths < 4:
        weights = (1 / hyp_lengths ,) * hyp_lengths
    else:
        weights = ( 1 / max_order ,) * max_order

    bp = brevity_penalty(ref_lengths, hyp_lengths)

    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False)
           for i, _ in enumerate(weights, start=1)]

    if smoothing_function is None:
        smoothing_function = SmoothingFunction().method0
    
    p_n = smoothing_function(p_n, references=references, hypothesis=hypothesis,
                             hyp_len=hyp_len, emulate_multibleu=emulate_multibleu)

    s = (w * math.log(p_i) for i, (w, p_i) in enumerate(zip(weights, p_n)))
    s =  bp * math.exp(math.fsum(s))
    return round(s, 4) if emulate_multibleu else s
        


def _ngrams(sequence, n):
    """
    simplify nltk.util.ngrams.
    Return the ngrams generated from a sequence of items, as an iterator.
    list(_ngrams(.., ..))
    """
    for i in range(0, len(sequence)-n+1):
        ngram = tuple(sequence[i:i+n])
        yield ngram

def modified_precision(references, hypothesis, n):
    """
    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hypothesis: A hypothesis translation.
    :type hypothesis: list(str)
    :param n: The ngram order.
    :type n: int
    :return: BLEU's modified precision for the nth order ngram.
    :rtype: Fraction
    """
    # Set an empty Counter if hypothesis is empty.
    counts = Counter(_ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    max_counts = {}
    for reference in references:  
        reference_counts = Counter(_ngrams(reference, n)) if len(reference) >= n else Counter()
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])  # 译文中出现的最大次数
    
    clipped_counts = {ngram:min(count, max_counts[ngram]) for ngram, count in counts.items()}

    numerator = sum(clipped_counts.values())
    denominator = max(1, sum(counts.values()))  # 1 to avoid zerodivision, result will be zero

    return Fraction(numerator, denominator, _normalize=False)


def closest_ref_length(references, hyp_len):
    ref_lens = (len(reference) for reference in references)
    closest_ref_len = min(ref_lens, key=lambda ref_len:
                          (abs(ref_len - hyp_len), ref_len))
    return closest_ref_len


def brevity_penalty(closest_ref_len, hyp_len):
    if hyp_len > closest_ref_len:
        return 1
    # If hypothesis is empty, brevity penalty = 0 should result in BLEU = 0.0
    elif hyp_len == 0:
        return 0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)

class SmoothingFunction:
    def __init__(self, epsilon=0.1, alpha=5, k=5):
        """
        :param epsilon: the epsilon value use in method 1
        :type epsilon: float
        :param alpha: the alpha value use in method 6
        :type alpha: int
        :param k: the k value use in method 4
        :type k: int
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = k

    def method0(self, p_n, *args, **kwargs):
        """
        No smoothing.
        """
        p_n_new = []
        _emulate_multibleu = kwargs['emulate_multibleu']  # boolean
        for i, p_i in enumerate(p_n):
            if p_i.numerator != 0:
                p_n_new.append(p_i)
            elif _emulate_multibleu and i < 5:  # ?
                return [sys.float.min]   # 不返回0，返回float.min
            else:
                _msg = str("\nCorpus/Sentence contains 0 counts of {}-gram overlaps.\n"
                           "BLEU scores might be undesirable; "
                           "use SmoothingFunction().").format(i+1)
                warnings.warn(_msg)
                # If this order of n-gram returns 0 counts, the higher order
                # n-gram would also return 0, thus breaking the loop here.
                break
        return p_n_new

    def method1(self, p_n, *args, **kwargs):  # 一样也是为了平滑0的情况?
        """
        add epsilon. 
        """
        return [(p_i.numerator + self.epsilon) / p_i.denominator   #其实可以不加p_i.numerator, 因为是0, 待测
            if p_i.numerator == 0 else p_i for p_i in p_n]

    def method2(self, p_n, *args, **kwargs):
        return [Fraction(p_i.numerator + 1, p_i.denominator + 1, _normalize=False) for p_i in p_n]

    def method3(self, p_n, *args, **kwargs):
        """
        Smoothing method 3: NIST geometric sequence smoothing
        The smoothing is computed by taking 1 / ( 2^k ), instead of 0, for each
        precision score whose matching n-gram count is null.
        k is 1 for the first 'n' value for which the n-gram match count is null/
        For example, if the text contains:
         - one 2-gram match
         - and (consequently) two 1-gram matches
        the n-gram count for each individual precision score would be:
         - n=1  =>  prec_count = 2     (two unigrams)
         - n=2  =>  prec_count = 1     (one bigram)
         - n=3  =>  prec_count = 1/2   (no trigram,  taking 'smoothed' value of 1 / ( 2^k ), with k=1)
         - n=4  =>  prec_count = 1/4   (no fourgram, taking 'smoothed' value of 1 / ( 2^k ), with k=2)
        """
        incvnt = 1  # From the mteval-v13a.pl, it's referred to as k.
        for i, p_i in enumerate(p_n):
            if p_i.numerator == 0:
                p_n[i] = 1 / (2 ** incvnt * p_i.denominator)  # 起码是比method2要小(为0时)
                incvnt += 1
        return p_n

    def method4(self, p_n, references, hypothesis, hyp_len, *args, **kwargs):
            
        for i, p_i in enumerate(p_n):
            if p_i.numerator == 0 and hyp_len != 0:  # 一旦 hyp_len为0的话(这样的话应该也没有p_n了吧？), p_n[i]还是为0, 考虑一下这种是什么情况下发生?
                incvnt = i + 1 * self.k / math.log(hyp_len) # Note that this K is different from the K from NIST.
                p_n[i] = 1 / incvnt
        return p_n

    def method5(self, p_n, references, hypothesis, hyp_len, *args, **kwargs):  
        # 不是太理解, 看论文吧 "http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf"
        """
        Smoothing method 5:
        The matched counts for similar values of n should be similar. To a  
        calculate the n-gram matched count, it averages the n−1, n and n+1 gram  
        matched counts.
        """
        m = {}
        p_n_plus = p_n + [modified_precision(references, hypothesis, 5)]
        m[-1] = p_n[0] + 1
        for i, p_i in enumerate(p_n):
            p_n[i] = (m[i-1] + p_i + p_n_plus[i+1]) / 3
            m[i] = p_n[i]
        return p_n

    def method6(self, p_n, references, hypothesis, hyp_len, *args, **kwargs):
        pass

    def method7(self, p_n, references, hypothesis, hyp_len, *args, **kwargs):
        pass
