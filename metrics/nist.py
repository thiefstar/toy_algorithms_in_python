# -*- coding:utf-8 -*-

# Python implementation of NIST  
# basic
# https://en.wikipedia.org/wiki/NIST_(metric)
# https://www.nltk.org/_modules/nltk/translate/nist_score.html 
# https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl
# http://www.cnblogs.com/by-dream/p/7765345.html
#
# nltk代码可能还有些问题, 需要再从论文考证
# (*modification?*)  一些小修改

import math
import fractions
from collections import Counter

try:
    fractions.Fraction(0, 1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    from nltk.compat import Fraction

def _ngrams(sequence, n):
    """
    simplify nltk.util.ngrams.
    Return the ngrams generated from a sequence of items, as an iterator.
    list(_ngrams(.., ..))
    """
    for i in range(0, len(sequence)-n+1):
        ngram = tuple(sequence[i:i+n])
        yield ngram

def sentence_nist(references, hypothesis, n=5):
    """
    :param references: reference sentences
    :type references: list(list(str))
    :param hypothesis: a hypothesis sentence
    :type hypothesis: list(str)
    :param n: highest n-gram order
    """
    return corpus_nist([references], [hypothesis], n)

def corpus_nist(list_of_references, hypotheses, n=5):
    """
    与bleu的区别，包括:
    sysoutput_lengths, bp, info
    """
    assert len(list_of_references) == len(hypotheses), "The number of hypotheses and their reference(s) should be the same"
    p_numerators = Counter() # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter() # Key = ngram order, and value = no. of ngram in ref.
    sysoutput_lengths = Counter() # Key = ngram order, and value = no. of ngram in hyp.
    hyp_lengths, ref_lengths = 0, 0

    for references, hypothesis in zip(list_of_references, hypotheses):
        for i, _ in enumerate(range(1, n+1)):  # i: 0->n-1  (表示n-1)
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator
            # sysoutput_lengths[i] += len(hypothesis) - (i - 1)  # sysout, n-gram number of w_i (=denominator? 但这边会出现<=0的情况)  
            sysoutput_lengths[i] += len(hypothesis) - i  #  (*modification1*) s_l[0] 应该为 unigram 片段数量, 待求证
 
        hyp_len =  len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)

    bp = nist_length_penalty(ref_lengths, hyp_lengths)
    
    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False)
           for i, _ in enumerate(range(1,n+1))]

    # Eqn 2 in Doddington (2002):
    # Info(w_1 ... w_n) = log_2 [ (# of occurrences of w_1 ... w_n-1) / (# of occurrences of w_1 ... w_n) ]
    # add by me. info , 对于一元词汇，分子的取值就是整个参考译文?的长度。(好像没表现出来，求证) 
    info = [0 if p_n[i].numerator == 0 or p_n[i+1].numerator == 0
            else math.log(p_n[i].numerator / p_n[i+1].numerator)
            for i in range(len(p_n)-1)]
    # print("+++")
    # print(info)
    # print(sysoutput_lengths)
    # print("+++")
    # return # 但是这边 info_i/sysoutput_lengths[i] 的对应关系还需考虑
    return sum(info_i/sysoutput_lengths[i] for i, info_i in enumerate(info) if not info_i == 0 ) * bp   # (*modification2*) 会出现 zerodivisionerror (如果hyp_len比较短而ref比较多, sysoutput_lengths可能为0或者负)
                                                                                                    # 但是虽然加了 "if not info_i == 0"， 程序的正确性需要验证。

def modified_precision(references, hypothesis, n):
    """
    same as bleu's modified_precision.
    """
    # Set an empty Counter if hypothesis is empty.
    counts = Counter(_ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    max_counts = {}
    for reference in references:  
        reference_counts = Counter(_ngrams(reference, n)) if len(reference) >= n else Counter()
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])  # 参考译文中出现的最大次数
    
    clipped_counts = {ngram:min(count, max_counts[ngram]) for ngram, count in counts.items()}

    numerator = sum(clipped_counts.values())  # (与参考译文共现的n-gram片段数量)
    denominator = max(1, sum(counts.values()))  # 1 to avoid zerodivision, result will be zero  (译文n-gram片段数目)

    return Fraction(numerator, denominator, _normalize=False)

def closest_ref_length(references, hyp_len):
    """
    same as bleu's closet_ref_length().
    """
    ref_lens = (len(reference) for reference in references)
    closest_ref_len = min(ref_lens, key=lambda ref_len:
                          (abs(ref_len - hyp_len), ref_len))
    return closest_ref_len

def nist_length_penalty(closest_ref_len, hyp_len):
    """
    Calculates the NIST length penalty, from Eq. 3 in Doddington (2002)

        penalty = exp( beta * log( min( len(hyp)/len(ref) , 1.0 )))

    where,

        `beta` is chosen to make the brevity penalty factor = 0.5 when the
        no. of words in the system output (hyp) is 2/3 of the average
        no. of words in the reference translation (ref)

    The NIST penalty is different from BLEU's such that it minimize the impact
    of the score of small variations in the length of a translation.
    See Fig. 4 in  Doddington (2002)

    惩罚因子


    """
    ratio = closest_ref_len / hyp_len
    if 0 < ratio < 1:
        ratio_x, score_x = 1.5, 0.5
        beta = math.log(score_x) / math.log(score_x) ** 2  # about -1.4427
        return math.exp(beta * math.log(ratio) ** 2)
    else:
        return max(min(ratio, 1.0), 0.0)
    