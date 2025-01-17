import pandas as pd
import csv
import pickle
import re
import string
import time
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

training = pd.read_csv('Airlinetweets.csv')

train_x = training['text']
train_y = training['type']
#print (training.groupby('type').count())

macronum=sorted(set(training['type']))
print (macronum)
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


#vectorizer = CountVectorizer()
#X = vectorizer.fit_transform(train_x)
tfidf_vectorizer=TfidfVectorizer(use_idf=True)
train_x=tfidf_vectorizer.fit_transform(train_x)

pickle.dump(tfidf_vectorizer, open('/tmp/vectorizer.sav', 'wb'))
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1, random_state=42)
#X_train = X_train.toarray()
#X_test = X_test.toarray()
#text_clf = Pipeline([
#    ('vect', CountVectorizer()),
#    ('tfidf', TfidfTransformer()),
#    ('clf', SVC(gamma='auto')),])
#clf = SVC(kernel='linear')
#clf = GaussianNB()
clf = LinearSVC(random_state=0, tol=1e-5)
#clf = SVC(kernel = 'linear', C = 1)
#t0 = time.time()
clf.fit(X_train, y_train)
#t1 = time.time()
predicted = clf.predict(X_test)
'''
if we want to see time taken for model train and prediction (from performance point of view)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
'''
#print (clf.score(X_test, y_test)) 

with open('/tmp/sgdclassifier.sav', 'wb') as f:
    pickle.dump(clf, f)

print("saved model!")
# and later you can load it

'''
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
text_clf = pickle.load(open('sgdclassifier.sav', 'rb'))
tweet = "sleep"
text_vector = vectorizer.transform([tweet])
result = text_clf.predict(text_vector)
print(result)
#print (text_clf.predict(tweet))
#print (text_clf.score(tweet, y_test))
'''