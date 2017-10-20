# -*- coding:utf-8 -*- 
# Author: Roc-J

import numpy as np
from nltk.corpus import brown

# 将文本分割成块
def splitter(data, num_words):
    words = data.split(' ')
    output = []

    cur_words = []
    cur_count = 0

    for word in words:
        cur_words.append(word)
        cur_count += 1

        if cur_count == num_words:
            output.append(' '.join(cur_words))
            cur_words = []
            cur_count = 0

    output.append(' '.join(cur_words))
    return output

if __name__ == '__main__':
    # 从布朗语料库中加载语料
    data = ' '.join(brown.words()[:10000])

    num_words = 1700
    text_chunks = splitter(data, num_words)
    print '\nThe numbers of the chunks is ', len(text_chunks)

