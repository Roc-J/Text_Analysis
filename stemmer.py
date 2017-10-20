# -*- coding:utf-8 -*- 
# Author: Roc-J

'''
提取文本数据的词干

'''
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

# 定义一些单词
words = ['table', 'probably', 'wolves', 'playing', 'is', 'dog', 'the', 'beaches', 'grounded', 'dreamt', 'envisions']
# 三种类别词干
stemmers = ['PORTER', 'LANCASTER', 'SNOWBALL']

stemmer_porter = PorterStemmer()
stemmer_lancaster = LancasterStemmer()
stemmer_snowball = SnowballStemmer('english')

# 格式化一行的输出
formatted_row = '{:>16}' * (len(stemmers) + 1)
print '\n', formatted_row.format('WORD', *stemmers), '\n'

# 输出单词
for word in words:
    stemmer_words = [stemmer_porter.stem(word), stemmer_lancaster.stem(word), stemmer_snowball.stem(word)]
    print formatted_row.format(word, *stemmer_words)