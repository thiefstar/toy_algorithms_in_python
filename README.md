├── edit_distance.py   # 两张Damerau-Levenshtein Distance的实现
├── hmm
│   ├── hmm.py  # hmm的学习算法实现: 监督与非监督学习
│   ├── prob_emit.pkl
│   ├── prob_start.pkl
│   ├── prob_trans.pkl
│   ├── __pycache__
│   ├── viterbi_hmm.py  # hmm预测算法实现: 近似算法与viterbi算法
│   └── word_seg.py  # 利用viterbi算法分词
├── metrics  # MT评测算法实现
│   ├── bleu.py
│   ├── nist.py
│   └── rouge.py
├── prime.py  # 素性测试与随机素数生成
├── priority_queue.py  # 优先队列实现(算法学习, 包括最大优先队列和最小索引优先队列)
├── __pycache__
├── substring_find.py  # 子字符串查找实现(算法学习, 包括KMP算法的next数组方式实现和DFA表实现, Boyer Moore算法, Rabin Karp算法)
└── tree.py  # 树的实现(算法学习, 包括Trie树, 三向单词查找树, 二叉查找树和红黑树)