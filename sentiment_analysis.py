# -*- coding:utf-8 -*- 
# Author: Roc-J

import numpy as np
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify import util

def extract_features(word_list):
    '''
    定义一个用于提取特征的函数
    :param word_list:
    :return:
    '''
    return dict([(word, True) for word in word_list])

if __name__ == '__main__':
    # 加载积极和消极评论
    positive_fileids = movie_reviews.fileids('pos')
    negative_fileids = movie_reviews.fileids('neg')

    feature_positive = [ (extract_features(movie_reviews.words(fileids=[f])), 'Positive') for f in positive_fileids]
    feature_negative = [ (extract_features(movie_reviews.words(fileids=[f])), 'Negative') for f in negative_fileids]

    # 分成训练集和测试集
    threshold_factor = 0.8
    threshold_positive = int(threshold_factor * len(feature_positive))
    threshold_negative = int(threshold_factor * len(feature_negative))

    feature_train = feature_positive[:threshold_positive] + feature_negative[:threshold_negative]
    feature_test = feature_positive[threshold_positive:] + feature_negative[threshold_negative:]

    print '\n Number of training datapoints: ', len(feature_train)
    print '\n Number of testing datapoints :', len(feature_test)

    # create model
    classifier = NaiveBayesClassifier.train(feature_train)

    print '\nAccuracy of the classifier : ', util.accuracy(classifier, feature_test)

    print '\n Top 10 most informative words: '
    for item in classifier.most_informative_features()[:10]:
        print item[0]

    # 输入一些简单的句子
    input_reviews = [
        "It is an amazing movie",
        "This is a dull movie. I would never recommend it to anyone.",
        "The cinematography is pretty great in this movie",
        "The direction was terrible and the story was all over the place"
    ]

    print '\nPredictions : '
    for review in input_reviews:
        print "\nReview: ", review
        probdist = classifier.prob_classify(extract_features(review.split()))
        pred_sentiment = probdist.max()

        print "predicted sentiment: ", pred_sentiment
        print 'Probability : ', round(probdist.prob(pred_sentiment), 2)