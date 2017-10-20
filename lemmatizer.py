# -*- coding:utf-8 -*- 
# Author: Roc-J

from nltk.stem import WordNetLemmatizer

# 定义一组单词来进行词性还原
words = ['table', 'probably', 'wolves', 'playing', 'is',
        'dog', 'the', 'beaches', 'grounded', 'dreamt', 'envision']

# 设置两个不同的词形还原器
lemmatizers = ['NOUN LEMMATIZER', 'VERB LEMMATIZER']

# create a lemmatizer
lemmatizer_wordnet = WordNetLemmatizer()

formatted_row = '{:>24}' * (len(lemmatizers) + 1)
print '\n', formatted_row.format('WORD', *lemmatizers), '\n'

for word in words:
    lemmatized_words = [lemmatizer_wordnet.lemmatize(word, pos='n'), lemmatizer_wordnet.lemmatize(word, pos='v')]

    print formatted_row.format(word, *lemmatized_words)