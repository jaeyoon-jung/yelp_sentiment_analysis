import json
import re
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
import nltk.classify.util
from nltk.stem.porter import PorterStemmer
import math
from collections import Counter
import pickle
import collections
import itertools
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def format(review):
    # compile regular expressions that match repeated characters and emoji unicode
    multiple = re.compile(r"(.)\1{1,}", re.DOTALL)
    # strip punctuation
    stripped = re.sub(r'[#|\!|\-|\+|:|//|\']', "", review)
    # strip numbers
    stripped = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', ' ', stripped).strip()
    # strip multiple occurrences of letters
    stripped = multiple.sub(r"\1\1", stripped)
    # strip all non-latin characters
    stripped = re.sub('[^a-zA-Z0-9|\']', " ", stripped).strip()
    # strip whitespace down to one.
    stripped = re.sub('[\s]+', ' ', stripped).strip()
    # upper case to lower case
    stripped = stripped.lower()

    return stripped


# construct feature vector
def word_feats(words):
    return dict([(word, True) for word in words])


# preprocess and extract features to classify sentiment of given text
def extract_feature(text, blacklist):
    formatted_text = format(text)
    stoplist = set(stopwords.words("english"))

    clean_review = [word for word in formatted_text.lower().split() if word not in
                    (list(stoplist) + blacklist) and len(word) > 1]
    feat = word_feats(clean_review)

    return feat

def bigram_word_feats(words, significant_uni, score_fn=BigramAssocMeasures.chi_sq, n=20):
    stoplist = set(stopwords.words("english"))
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams) if ngram not in significant_uni and
                ngram not in stoplist])

# construct a training dataset and train a model on it
def main():
    # build a training dataset
    positives = []
    negatives = []

    yelp_f = 'yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json'
    f = open(yelp_f, 'r')
    for i in range(0, 500000):
        review = json.loads(f.readline())
        review_text = review['text']
        if review['stars'] == 5:
            positives.append(review_text)
        elif review['stars'] == 1:
            negatives.append(review_text)
    
    print ('Collected %d positive reviews' %len(positives))
    print ('Collected %d negative reviews' %len(negatives))

    # stoplist from nltk
    stoplist = set(stopwords.words("english"))

    # lists to store preprocessed text
    posreviews = []
    negreviews = []

    # remove stopword and extract stemword
    for positive in positives:
        formatted_positive = format(positive)
        pos = [PorterStemmer().stem(word).encode("utf-8") for word in formatted_positive.lower().split() if word not in
               stoplist and len(word) > 1]
        posreviews.append(pos)

    for negative in negatives:
        formatted_negative = format(negative)
        neg = [PorterStemmer().stem(word.encode("utf-8")) for word in formatted_negative.lower().split() if word not in
               stoplist and len(word) > 1]
        negreviews.append(neg)

    # identify words that appear too often in both positive and negative reviews
    # remove them since they are probably not very informative
    poswords = [word for wordlist in posreviews for word in wordlist]
    poscnt = Counter()
    for posword in poswords:
        poscnt[posword] += 1

    top1000_pos = poscnt.most_common(1000)

    negwords = [word for wordlist in negreviews for word in wordlist]
    negcnt = Counter()
    for negword in negwords:
        negcnt[negword] += 1

    top1000_neg = negcnt.most_common(1000)

    freq_pos = []
    freq_neg = []

    # top1000_pos and top1000_neg are dictionary pairs of word and number of appearance.
    # Extract just the words (key)
    for pos_dict in top1000_pos:
        pos_key = pos_dict[0]
        freq_pos.append(pos_key)
    for neg_dict in top1000_neg:
        neg_key = neg_dict[0]
        freq_neg.append(neg_key)

    # application of zipfian theory- words of top100 frequency shold be good enough
    toocommon = [common for common in freq_pos if common in freq_neg]

    for i in range(0, len(posreviews)):
        posreviews[i] = [word for word in posreviews[i] if word not in toocommon and len(word) > 1]

    for i in range(0, len(negreviews)):
        negreviews[i] = [word for word in negreviews[i] if word not in toocommon and len(word) > 1]

    # extract features
    negfeats = [(word_feats(review), 'neg') for review in negreviews]
    posfeats = [(word_feats(review), 'pos') for review in posreviews]

    poscutoff = int(len(posfeats) * 0.8)
    negcutoff = int(len(negfeats) * 0.8)
    pos_n = len(posfeats)
    neg_n = len(negfeats)

    # build a naive bayes classifier
    trainpos = posfeats[0:poscutoff]
    trainneg = negfeats[0:negcutoff]
    testpos = posfeats[poscutoff:pos_n]
    testneg = negfeats[negcutoff:neg_n]
    trainfeats = trainneg + trainpos
    naive_classifier = NaiveBayesClassifier.train(trainfeats)
    naive_classifier.show_most_informative_features()

    testfeats = testneg + testpos

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        predicted = naive_classifier.classify(feats)
        testsets[predicted].add(i)

    # 86.7% accuracy.
    print ('accuracy:', nltk.classify.util.accuracy(naive_classifier, testfeats))
    print ('pos precision:', nltk.precision(refsets['pos'], testsets['pos']))
    print ('pos recall:', nltk.recall(refsets['pos'], testsets['pos']))
    print ('neg precision:', nltk.precision(refsets['neg'], testsets['neg']))
    print ('neg recall:', nltk.recall(refsets['neg'], testsets['neg']))

    # save unigram classifier
    pickle.dump(naive_classifier, open("unigram_classifier.p", "wb"))

    # now bigram analysis!
    # lists to store preprocessed text
    bi_posreviews = []
    bi_negreviews = []

    # remove stopword and get extract stemword
    for positive in positives:
        formatted_positive = format(positive)
        pos = [PorterStemmer().stem(word) for word in formatted_positive.lower().split() if len(word) > 1]
        bi_posreviews.append(pos)

    for negative in negatives:
        formatted_negative = format(negative)
        neg = [PorterStemmer().stem(word) for word in formatted_negative.lower().split() if len(word) > 1]
        bi_negreviews.append(neg)

    # extract features
    bi_negfeats = [(bigram_word_feats(words=review, significant_uni=toocommon, n=20), 'neg') for review in bi_negreviews
                   if len(set(review)) > 1]
    bi_posfeats = [(bigram_word_feats(words=review, significant_uni=toocommon, n=20), 'pos') for review in bi_posreviews
                   if len(set(review)) > 1]

    poscutoff = int(len(bi_posfeats) * 0.8)
    negcutoff = int(len(bi_negfeats) * 0.8)
    pos_n = len(bi_posfeats)
    neg_n = len(bi_negfeats)

    # build a naive bayes classifier
    bi_trainpos = bi_posfeats[0:poscutoff]
    bi_trainneg = bi_negfeats[0:negcutoff]
    bi_testpos = bi_posfeats[poscutoff:pos_n]
    bi_testneg = bi_negfeats[negcutoff:neg_n]
    bi_trainfeats = bi_trainneg + bi_trainpos
    bigram_naive_classifier = NaiveBayesClassifier.train(bi_trainfeats)
    bigram_naive_classifier.show_most_informative_features()

    # 83.9% accuracy. not bad!
    bi_testfeats = bi_testneg + bi_testpos
    bi_refsets = collections.defaultdict(set)
    bi_testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(bi_testfeats):
        bi_refsets[label].add(i)
        bi_predicted = bigram_naive_classifier.classify(feats)
        bi_testsets[bi_predicted].add(i)

    # 90.8% accuracy
    print ('accuracy:', nltk.classify.util.accuracy(bigram_naive_classifier, bi_testfeats))
    print ('pos precision:', nltk.precision(bi_refsets['pos'], bi_testsets['pos']))
    print ('pos recall:', nltk.recall(bi_refsets['pos'], bi_testsets['pos']))
    print ('neg precision:', nltk.precision(bi_refsets['neg'], bi_testsets['neg']))
    print ('neg recall:', nltk.recall(bi_refsets['neg'], bi_testsets['neg']))

    # save the model in a pickle file
    pickle.dump(bigram_naive_classifier, open("bigram_classifier.p", "wb"))

    # save frequent word list
    pickle.dump(toocommon, open("uninformative.p", 'wb'))