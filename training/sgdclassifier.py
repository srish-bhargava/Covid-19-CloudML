import pandas as pd
import csv
import pickle
import re
import string
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

training = pd.read_csv('Airlinetweets.csv')

train_x = training['text']
train_y = training['type']

macronum=sorted(set(training['type']))
#print (macronum)
macro_to_id = dict((note, number) for number, note in enumerate(macronum))

def fun(i):
    return macro_to_id[i]

train_y=training['type'].apply(fun)


texts = []
for tweet in training['text']:
    translator=str.maketrans('','',string.punctuation)
    tweet=tweet.translate(translator)
    tweet = tweet.lower() # convert text to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', tweet) # remove URLs
    tweet = re.sub('@[^\s]+', '', tweet) # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) 
    #tweet = remove_stopwords(tweet)
    #print (tweet)
    texts.append(tweet)

train_x = texts

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1, random_state=42)
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC(gamma='auto')),])

text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test)

print (text_clf.score(X_test, y_test)) 

with open('sgdclassifier.pkl', 'wb') as f:
    pickle.dump(text_clf, f)

# and later you can load it
#with open('filename.pkl', 'rb') as f:
#    text_clf = pickle.load(f)
#to make an inference