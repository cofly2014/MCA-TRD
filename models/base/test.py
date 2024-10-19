import numpy as np
from gensim.downloader import load
# 下载并加载预训练的Word2Vec模型

from gensim.models import KeyedVectors
from gensim.test.utils import datapath
wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/home/guofei/gensim-data/word2vec-google-news-300/GoogleNews-vectors-negative300.bin"), binary=True)


#wv = load('word2vec-google-news-300')

# 检查词汇表中的前10个词
'''
for i, word in enumerate(wv_from_bin.index_to_key):
    if i <= 10:
        print(word)
'''

# 打印词汇表总数
print(len(wv_from_bin.index_to_key))

# 获取指定词的词向量
word = 'king'
vector = wv_from_bin[word]
print(f'The vector for word "{word}" has shape: {vector.shape}')

# 计算两个词的相似度
word1 = 'man'
word2 = 'woman'
similarity = wv_from_bin.similarity(word1, word2)
print(f'The similarity between {word1} and {word2} is: {similarity}')

# 查找与给定词最相似的词
similar_words = wv_from_bin.most_similar(word, topn=5)
print(f'Most similar words to {word}: {similar_words}')