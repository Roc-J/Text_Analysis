# -*- coding:utf-8 -*- 
# Author: Roc-J

import numpy as np
from nltk.corpus import brown
from chunking import splitter
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    data = ' '.join(brown.words()[:10000])

    # 每块包含的单词数量
    num_words = 2000

    chunks = []
    counter = 0
    text_chunks = splitter(data, num_words)
    for text in text_chunks:
        chunk = {'index':counter, 'text':text}
        chunks.append(chunk)
        counter += 1

    # 提取文档-词矩阵
    vectorizer = CountVectorizer(min_df=5, max_df=.95)
    doc_term_matrix = vectorizer.fit_transform([chunk['text'] for chunk in chunks])

    # 从vectorizer对象中提取词汇
    vocab = np.array(vectorizer.get_feature_names())
    print '\nVocabulary: '
    print vocab

    print '\nDocment term matrix: '
    chunk_names = ['Chunk-0', 'Chunk-1', 'Chunk-2', 'Chunk-3', 'Chunk-4']
    formated_row = '{:>12}' *(len(chunk_names) + 1)
    print '\n', formated_row.format('Word', *chunk_names), '\n'

    for word, item in zip(vocab, doc_term_matrix.T):
        # 'item' 是压缩的稀疏矩阵
        output = [str(x) for x in item.data]
        print formated_row.format(word, *output)