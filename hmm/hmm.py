# -*- coding:utf-8 -*-

# basic
# 统计学习方法
# http://www.hankcs.com/ml/hidden-markov-model.html
# www.52nlp.cn/Itenyh版-用HMM做中文分词四：a-pure-hmm-分词器

# todo 测试部分可以清除

import numpy as np
import collections
import pickle


def build_dataset():
    """
    构建数据集, 
    (词频低于<?>的考虑过滤,用<unk>代替,看看效果会好一些吗)
    :param dictionary: word2idx
    :param reverse_dictionary: idx2word
    """
    # <unk> - -1
    # <none> - 0 
    return

class HMM(object):
    """
    Order 1 Hidden Markov Model
 
    Attributes
    ----------
    A : numpy.ndarray
        State transition probability matrix
    :type A: (states_len, state_len)
    B: numpy.ndarray
        Output emission probability matrix with shape(N, number of output types)
    :type B: (states_len, obs_len)
    pi: numpy.ndarray
        Initial state probablity vector
    :type pi: (1, states_len).T
    """
 
    def __init__(self, A, B, pi):
        # init value
        self.A = A
        self.B = B
        self.pi = pi

    def simulate(self, T):

        def draw_from(probs):
            return np.where(np.random.multinomial(1,probs) == 1)[0][0]
     
        observations = np.zeros(T, dtype=int)
        states = np.zeros(T, dtype=int)
        states[0] = draw_from(self.pi)
        observations[0] = draw_from(self.B[states[0],:])
        for t in range(1, T):
            states[t] = draw_from(self.A[states[t-1],:])
            observations[t] = draw_from(self.B[states[t],:])
        return observations,states

    def _forward(self, obs_seq):
        """
        前向算法 alpha_t(i)
        :type F: 如果是一条观测序列的话 (states_len, obs_len)
        """
        N = self.A.shape[0]
        T = len(obs_seq)
    
        F = np.zeros((N,T))
        F[:,0] = self.pi * self.B[:, obs_seq[0]]
    
        for t in range(1, T):
            for n in range(N):
                F[n,t] = np.dot(F[:,t-1], (self.A[:,n])) * self.B[n, obs_seq[t]]

        return F

    def _backward(self, obs_seq):
        """
        后向算法 beta_t(i)
        :type X: (states_len, obs_len)
        """
        N = self.A.shape[0]
        T = len(obs_seq)
     
        X = np.zeros((N,T))
        X[:,-1:] = 1
     
        for t in reversed(range(T-1)):
            for n in range(N):
                X[n,t] = np.sum(X[:,t+1] * self.A[n,:] * self.B[:, obs_seq[t+1]])
     
        return X

    # 监督学习方法(统计)
    def stati_calculate(self, save=False):
        """
        利用极大似然估计法来估计HMM参数
        input: text file, every line: word/[BEMS], split with a blank space.
        return: None (with updating of A, B, pi)
        """
        pass

    # 非监督学习方法
    def baum_welch_train(self, observations, criterion=0.05):
        """
        :param observations: 观测序列
        :type observations: numpy
        :param criterion:
        """
        n_states = self.A.shape[0]
        n_samples = len(observations)

        done = False
        while not done:
            # Initialize alpha, 前向概率, (n_states, n_samples)
            alpha = self._forward(observations)
            # Initialize beta, 后向概率
            beta = self._backward(observations)

            xi = np.zeros((n_states, n_states, n_samples-1))  # t: 1 ~ T-1
            for t in range(n_samples-1):
                denom = np.dot(np.dot(alpha[:,t].T, self.A) * self.B[:,observations[t+1]].T, beta[:,t+1])
                for i in range(n_states):
                    numer = alpha[i,t] * self.A[i,:] * self.B[:,observations[t+1]].T * beta[:,t+1].T
                    xi[i,:,t] = numer / denom

            # gamma_t(i) = P(q_t = S_i | O, hmm)
            gamma = np.sum(xi,axis=1)  # shape=()
            # Need final gamma element for new B
            prod =  (alpha[:,n_samples-1] * beta[:,n_samples-1]).reshape((-1,1))
            gamma = np.hstack((gamma,  prod / np.sum(prod))) # append one more to gamma!(cuz one prod less)

            newpi = gamma[:,0]
            newA = np.sum(xi, 2) / np.sum(gamma[:,:-1], axis=1).reshape((-1,1))

            newB = np.copy(self.B)
            num_levels = self.B.shape[1]
            sum_gamma = np.sum(gamma,axis=1)  # t : 1 ~ T
            for lev in range(num_levels):
                mask = observations == lev  # gamma when o_t == v_k  -> b_j(k),  gamma.shape=(state_len, obs_len)
                newB[:,lev] = np.sum(gamma[:,mask], axis=1) / sum_gamma  

            if np.max(abs(self.pi - newpi)) < criterion and \
                            np.max(abs(self.A - newA)) < criterion and \
                            np.max(abs(self.B - newB)) < criterion:
                done = True
            print(".", end=" ")
            self.A[:],self.B[:],self.pi[:] = newA, newB, newpi

if __name__ == '__main__':

    from viterbi_hmm import viterbi

    h = HMM(np.array([[ 0.7,0.3],
                    [0.4,0.6]]),
            np.array([[ 0.5,0.4,0.1],
                    [ 0.1,0.3,0.6]]),
            np.array([0.6, 0.4]))

    observations_data, states_data = h.simulate(100)
    print(observations_data)
    print(states_data)

    guess = HMM(np.array([[0.5, 0.5],
                          [0.5, 0.5]]),
                np.array([[0.3, 0.3, 0.3],
                          [0.3, 0.3, 0.3]]),
                np.array([0.5, 0.5])
    )

    guess.baum_welch_train(observations_data)
    print()
    print(h.pi)
    print(h.A)
    print(h.B)
    prob, path = viterbi(observations_data, [0, 1], h.pi, h.A, h.B)
    p = 0.0
    print(path)
    for i in range(len(path)):
        if path[i] == states_data[i]:
            p += 1
    print(p / len(states_data))
        
    