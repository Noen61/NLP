MAX_VOCAB_SIZE = 30000
C= 3 #周围单词个数（context window）
K = 100 #下采样（number of negative samoles）
NUM_EPOCHS =  100 #迭代次数
BATCH_SIZE = 512 #批样本数
LEARNING_RATE = 0.02 #学习率
EMBEDDING_SIZE = 100 #词向量长度