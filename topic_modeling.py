# -*- coding:utf-8 -*- 
# Author: Roc-J

from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora
from nltk.corpus import stopwords

def load_data(input_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data.append(line[:-1])

    return data

class Preprocessor(object):
    '''
    预处理文本的类
    '''
    def __init__(self):
        # 创建正则表达式解析器
        self.tokenizer = RegexpTokenizer(r'\w+')

        # 获取停用词
        self.stop_words_english = stopwords.words('english')

        # 创建词干提取器
        self.stemmer = SnowballStemmer('english')

    def process(self, input_text):
        tokens = self.tokenizer.tokenize(input_text.lower())

        tokens_stopwords = [x for x in tokens if x not in self.stop_words_english]

        tokens_stemmed = [self.stemmer.stem(x) for x in tokens_stopwords]

        return tokens_stemmed

if __name__ == '__main__':
    input_file = 'data_topic_modeling.txt'

    data = load_data(input_file)

    preprocessor = Preprocessor()

    # 创建一组经过预处理的文档
    processed_tokens = [preprocessor.process(x) for x in data]
    # 创建基于标记文档的词典
    dict_tokens = corpora.Dictionary(processed_tokens)

    corpus = [dict_tokens.doc2bow(text) for text in processed_tokens]

    # 假定文本可以分成两个主题，将使用隐含狄利克雷分布（LDA）做主题建模
    num_topics = 2
    num_words = 4

    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dict_tokens, passes=25)

    print '\nMost contributing words to the topics: '
    for item in ldamodel.print_topics(num_topics=num_topics, num_words=num_words):
        print '\nTopic ', item[0], '==>', item[1]