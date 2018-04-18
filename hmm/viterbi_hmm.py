# -*- coding:utf-8 -*-

# basic
# 统计学习方法
# https://github.com/hankcs/Viterbi/blob/master/src/com/hankcs/algorithm/Viterbi.java
# http://www.hankcs.com/nlp/hmm-and-segmentation-tagging-named-entity-recognition.html

# todo:使用np
# todo:hanlp 的标准实现, 再看一下

# 还有一个近似算法, 对HMM预测 (统计学习方法)
def sample_pred(obs, states, start_p, trans_p, emit_p):
    """
    :param obs: 观测序列 o_t
    :type obs: list
    :param states: 状态集合
    :param start_p: 初始状态概率
    :param trans_p: 状态转移概率
    :param emit_p: 发射概率, 观测概率
    """
    T = len(obs)  # T 是观测序列长
    # 记录t时间, state状态的前向概率
    alpha = [{}]
    # 记录t时间, state状态的后向概率
    beta = [{} for _ in range(T)]
    # result
    path = []
    # 初始化, t = 0
    for y in states:
        alpha[0][y] = start_p[y] * emit_p[y][obs[0]]
        beta[T-0-1][y] = 1
    # 递推, t : 1 -> N-1
    for t in range(1, T):
        alpha.append({})
        t_b = T - t - 1
        for y in states:
            alpha[t][y] = sum([alpha[t-1][y_] * trans_p[y_][y] for y_ in states]) * emit_p[y][obs[t]]
            beta[t_b][y] = sum([trans_p[y][y_] * emit_p[y_][obs[t_b+1]] * beta[t_b+1][y_] for y_ in states])

    for t in range(T):
        # denominator for calculating gamma_t(state)
        denominator = sum([alpha[t][y_] * beta[t][y_] for y_ in states])
        (gamma_t, state) = max((alpha[t][y] * beta[t][y] / denominator, y) for y in states)
        path.append(state)

    return path

def viterbi(obs, states, start_p, trans_p, emit_p):
    """
    状态, 观测 (= 隐状态, 显状态)
    :param obs: 观测序列 o_t
    :type obs: list
    :param states: 状态集合
    :param start_p: 初始状态概率
    :param trans_p: 状态转移概率
    :param emit_p: 发射概率, 观测概率
    """
    # 记录t时间, state状态的路径概率
    V = [{}]
    # 记录到状态state的最优路径 
    path = {}

    # 初始化, t = 0
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]
    
    # 递推, t : 1 -> N-1
    for t in range(1, len(obs)):
        V.append({})
        new_path = {}

        for y in states:
            # y_ is (t-1) states, state_ is the argmax y_
            (prob, state_) = max([(V[t-1][y_] * trans_p[y_][y] * emit_p[y][obs[t]], y_) for y_ in states])
            V[t][y] = prob
            new_path[y] = path[state_] + [y]
        
        path = new_path

    (max_prob, state) = max([(V[len(obs)-1][y], y) for y in states])
    return (max_prob, path[state])


if __name__ == '__main__':

    import numpy as np

    def example():
        # states = ('Rainy', 'Sunny')
        # observations = ('walk', 'shop', 'clean')
        # start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
        # transition_probability = {
        #     'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
        #     'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
        #     }
        # emission_probability = {
        #     'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
        #     'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
        # }
        
        #   隐状态
        hidden_state = ['sunny', 'rainy']
        #   观测序列
        obsevition = ['walk', 'shop', 'clean']
        states = [0, 1]
        observations = [0, 1, 2]
        #   初始状态，测试集中，0.6概率观测序列以sunny开始
        start_probability = [0.6, 0.4]
        #   转移概率，0.7：sunny下一天sunny的概率
        transition_probability = np.array([[0.7, 0.3], [0.4, 0.6]])
        #   发射概率，0.4：sunny在0.4概率下为shop
        emission_probability = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])

        # sample, viterbi
        return viterbi(observations,
                    states,
                    start_probability,
                    transition_probability,
                    emission_probability)
    
    print(example())