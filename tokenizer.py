# -*- coding:utf-8 -*- 
# Author: Roc-J

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer

text = "Are you curious about tokenization? Let's see how it works! We need to analyze a couple of sentences with punctuations to see it in action."

# 句子解析器
sent_tokenize_list = sent_tokenize(text)
print "\nSentence tokenizer:"
print sent_tokenize_list

# 单词解析器
print "\nWord tokenizer: "
print word_tokenize(text)

# 需要将标点符号保留到不同的句子标记中
word_punct_tokenize = WordPunctTokenizer()
print "\nWord punct tokenizer: "
print word_punct_tokenize.tokenize(text)
