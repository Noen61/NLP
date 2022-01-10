from string import punctuation
from collections import Counter
import pandas as pd
import numpy as np
import random
import math
from const import MAX_VOCAB_SIZE



def build_vocab(path):
    punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
    with open(path,"r") as fin:
        text = fin.read()
    text = re.sub(r"[{}]+".format(punc)," ",text)
    text = text.split()
    #构造词频字典
    vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
    #未知词
    vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
    #建立词和索引的对应
    idx_to_word = [word for word in vocab.keys()]
    word_to_idx = {word:i for i,word in enumerate(idx_to_word)}
    #计算词频,按照原论文转换为3/4次方
    word_counts = np.array([count for count in vocab.values()],dtype=np.float32)
    word_freqs = word_counts/np.sum(word_counts)
    word_freqs = word_freqs ** (3./4.)
    word_freqs = word_freqs / np.sum(word_freqs)
    VOCAB_SIZE = len(idx_to_word) 

    return idx_to_word,word_to_idx,word_freqs,VOCAB_SIZE