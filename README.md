# 自然语言处理基本任务

包含词向量训练、文本分类、文本匹配、机器翻译、关系抽取和事件抽取等自然语言处理基本任务。

## 1.[词向量训练](https://github.com/Noen61/NLP/tree/master/Word2vec)

使用pytorch深度学习框架，利用已经分好词的，给定的中文和英文语料训练对应的词向量。使用的训练词向量的模型为基于负采样（negative sampling）的skip-gram模型。

## 2.[序列标注](https://github.com/Noen61/NLP/tree/master/Sequence%20Tagging)

使用pytorch深度学习框架，利用data目录下的中文文本数据训练模型来完成序列标注任务（命名实体识别），识别出文本中的人名、地名和组织机构名。序列标注模型采用神经网络+条件随机场Bi-LSTM+CRF模型实现。

## 3.[关系抽取](https://github.com/Noen61/NLP/tree/master/Relation%20Extraction)

使用pytorch深度学习框架建立一个关系抽取模型，实现在英文数据集上进行的关系抽取，识别出文本中的人名、地名和组织机构名。参照论文”Matching the Blanks: Distributional Similarity for Relation Learning”，以Bert模型为基础设计一个通用的关系抽取器。

## 4.[事件抽取](https://github.com/Noen61/NLP/tree/master/Event%20Extraction)

使用pytorch深度学习框架建立一个事件抽取模型，实现在中文数据集上进行的事件抽取，识别出文本中的事件及其对应的触发词，与相应的事件角色及其对应的论元。把事件抽取当成两阶段任务，先做事件触发词检测，再做事件论元检测，每阶段都是当作一个序列标注任务；采用pipeline模型，网络的主体架构采用BERT+BiLSTM来分别进行触发词的识别和论元角色的分配。