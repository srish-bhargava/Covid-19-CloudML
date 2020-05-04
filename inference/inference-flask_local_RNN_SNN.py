"""Deploying imagenet sample model"""

import os
import io
import flask
import json
import keras
import keras.preprocessing.text as kpt
from keras.models import model_from_json
import numpy as np
from keras.preprocessing.text import Tokenizer
from flask import Flask, jsonify, request

app = flask.Flask(__name__)

tokenizer = Tokenizer(num_words=3000)

# for human-friendly printing
labels = ['negative', 'neutral','positive']
with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            print("'%s' not in training corpus; ignoring." %(word))
    return wordIndices

# def sgd_model():
#     train_x = pd.read_csv('transform.csv')
#     tfidf_vectorizer=TfidfVectorizer(use_idf=True)
#     train_x=tfidf_vectorizer.fit_transform(train_x)
#     with open('sgdclassifier.pkl', 'wb') as f:
#         pickle.dump(clf, f)
#     # and later you can load it
#     with open('sgdclassifier.pkl', 'rb') as f:
#         text_clf = pickle.load(f)
#     # tweet = "play"
#     tweet=tfidf_vectorizer.transform([tweet])
#     #tweet = tweet.toarray()
#     labels = [0,1,2]
#     print (text_clf.predict(tweet))
#     return flask.jsonify(text_clf.predict(tweet)

# read in your saved model structure
# JSON FOR SNN
# json_file = open('model.json', 'r')
# JSON for RNN
# json_file = open('model_rnn.json', 'r')


# loaded_model_json = json_file.read()
# json_file.close()

# and create a model from that
# model = model_from_json(loaded_model_json)

# and weight your nodes with your saved values
# MODEL FOR SNN
# model.load_weights('model1.h5')
#model for RNN
# model.load_weights('model_rnn.h5')

#curl -i -X PUT -F name=Test -F filedata=@SomeFile.pdf "http://localhost:5000/"
@app.route("/")
def predict():
    data = {"success": False}
    model_number=request.args['model_number']
    print("model number is " + model_number)
    tweet = request.args['name']
    
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # and create a model from that
    model = model_from_json(loaded_model_json)
    model.load_weights('model1.h5')


    if model_number=="1":
        # sgd_model()
        # JSON for RNN
        json_file = open('model_rnn.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        #model for RNN
        model = model_from_json(loaded_model_json)
        model.load_weights('model_rnn.h5')


    # JSON FOR SNN


    # ensure an image was properly uploaded to our endpoint
    #if flask.request.method == "POST":
            
    #check if tweet length is 0, break out


    testArr = convert_text_to_index_array(tweet)
    input_value = tokenizer.sequences_to_matrix([testArr], mode='binary')
    pred = model.predict(input_value)
    print("%s sentiment; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
    data = {}
    data["predictions"] = str(labels[np.argmax(pred)])
    data["confidence"] = str(pred[0][np.argmax(pred)] * 100) + " %"
    print(data)
    return flask.jsonify(data)

if __name__ == '__main__':
    print("Loading model and starting Flask Server..")
    #load_model()
    app.run(host='0.0.0.0', port=8001, threaded=False)