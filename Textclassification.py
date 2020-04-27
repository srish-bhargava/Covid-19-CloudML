import re
import pandas as pd
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords
#from textblob import TextBlob 
#from nltk.tokenize.casual import TweetTokenize
#from os import path
#from PIL import Image
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

tweets = pd.read_excel("../text_classification_dataset.xlsx")
cleaned_tweets = []
for tweet in tweets.text:
    tweet = tweet.lower() # convert text to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) 
    #print (tweet)
    cleaned_tweets.append(tweet)

text_clf = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', MultinomialNB()),
])
text_clf.fit(tweets['text'], tweets['type'])
X_train, X_test, y_train, y_test = train_test_split(tweets['text'], tweets['type'], test_size=0.1, random_state=42)
predicted = text_clf.predict(X_test)
print (np.mean(predicted == y_test))