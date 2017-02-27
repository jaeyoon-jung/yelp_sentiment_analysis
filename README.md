# Analyzing Sentiment of Yelp Reviews

* Objective: build sentiment analysis model that can classify the sentiment of Yelp reviews and derive interesting insight from the result.  
* Data: ```yelp_academic_dataset_review.json``` from Yelp Dataset Challenge. (https://www.yelp.com/dataset_challenge)

## Training Data:
Used 1 starred and 5 starred reveiws from the first 500,000 lines of reviews, assuming that they contain negative and positive sentiment respectively. 

## Preprocessing
The reviews are tokenized after the following formatting:
* All characters to lower case
* Repeated characters are reduced to one. This is to prevent having words like 'gooooooood'
* Emojis are stripped
* Punctuations are stripped
* Non-Latin characters are stripped
* All white space is reduced to one
* All numbers are stripped
* English stopwords (list obtained from NLTK module) are stripped
* All words are reducted to their stemmers, using Porter Stemmer Algorithm

Then, I also removed words that appear in both classes too frequently, in order to reduce the number of uninformative features

## Final Model: 
Naive Bayes Classifier with Unigram and 20 Significant Bigram Features
* Accuracy: 0.9080
* Positive Precision: 0.9940
* Positive Recall: 0.8810
* Negative Precision: 0.7741
* Negative Recall: 0.9850
