import json
import re
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
import nltk.classify.util
import math
from collections import Counter
import pickle


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


# read lines between start and end and classify its sentiment
def save_SA_results(yelp_f, output_f, start, end, model, blacklist):
    fp = open(yelp_f)
    for i, line in enumerate(fp):
        if i in range(start, end):
            review = json.loads(line)
            with open(output_f, 'a') as outfile:
                feat = extract_feature(review['text'], blacklist)
                sentiment_dist = model.prob_classify(feat)

                sentiment = model.classify(feat)
                pos_prob = sentiment_dist.prob('pos')
                neg_prob = sentiment_dist.prob('neg')

                stars = review['stars']

                data = {}
                data['date'] = review['date']
                data['text'] = review['text']
                data['votes'] = review['votes']
                data['stars'] = review['stars']
                data['sentiment'] = sentiment
                data['positive_probability'] = pos_prob
                data['negative_probability'] = neg_prob

                out = json.dumps(data)
                outfile.write(out + '\n')
    print ('Wrote ' + output_f)


# construct a training dataset and train a model on it
def main():
    # build a training dataset
    positives = []
    negatives = []

    yelp_f = 'yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json'
    start = 0
    end = 250000
    fp = open(yelp_f)
    for i, line in enumerate(fp):
        if i in range(start, end):
            review = json.loads(line)
            review_text = review['text']

            if review['stars'] == 5:
                positives.append(review_text)
            elif review['stars'] == 1:
                negatives.append(review_text)

    positives = list(set(positives))
    negatives = list(set(negatives))

    print ('Positive training data: ' + str(len(positives)) + " reviews")
    print ('Negative training data: ' + str(len(negatives)) + " reviews")

    # stoplist from nltk
    stoplist = set(stopwords.words("english"))

    posreviews = []
    for positive in positives:
        formatted_positive = format(positive)
        pos = [word for word in formatted_positive.lower().split() if word not in
               stoplist and len(word) > 1]
        posreviews.append(pos)

    negreviews = []
    for negative in negatives:
        formatted_negative = format(negative)
        neg = [word for word in formatted_negative.lower().split() if word not in
               stoplist and len(word) > 1]
        negreviews.append(neg)

    # identify words that appear too often in both positive and negative reviews
    # remove them since they are probably not very informative
    poswords = [item for sublist in posreviews for item in sublist]
    poscnt = Counter()
    for posword in poswords:
        poscnt[posword] += 1

    top1000_pos = poscnt.most_common(1000)

    negwords = [item for sublist in negreviews for item in sublist]
    negcnt = Counter()
    for negword in negwords:
        negcnt[negword] += 1

    top1000_neg = negcnt.most_common(1000)

    zipf_pos = []
    zipf_neg = []
    for pos_dict in top1000_pos:
        pos_key = pos_dict[0]
        zipf_pos.append(pos_key)
    for neg_dict in top1000_neg:
        neg_key = neg_dict[0]
        zipf_neg.append(neg_key)

    # application of zipfian theory- words of top1000 frequency shold be good enough
    toocommon = [val for val in zipf_pos if val in zipf_neg]

    for i in range(0, len(posreviews)):
        posreviews[i] = [word for word in posreviews[i] if word not in toocommon and len(word) > 1]

    for i in range(0, len(negreviews)):
        negreviews[i] = [word for word in negreviews[i] if word not in toocommon and len(word) > 1]

    # extract features
    negfeats = [(word_feats(review), 'neg') for review in negreviews]
    posfeats = [(word_feats(review), 'pos') for review in posreviews]

    poscutoff = math.ceil(len(posfeats) * 0.8)
    negcutoff = math.ceil(len(negfeats) * 0.8)
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

    # 83.9% accuracy. not bad!
    testfeats = testneg + testpos
    print ('Model Accuracy:', nltk.classify.util.accuracy(naive_classifier, testfeats))

    # save the model in a pickle file
    pickle.dump(naive_classifier, open("classifier.p", "wb"))

    # divide the original dataset into 10 chunks, save the result fo each
    for i in range(0, 10):
        result_f = 'naive_result/result' + str(i) + '.json'
        start = 0 + (i * 268506)
        end = 268506 * (i + 1)
        save_SA_results(yelp_f, result_f, start, end, naive_classifier, toocommon)
