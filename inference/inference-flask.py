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

# read in your saved model structure
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# and create a model from that
model = model_from_json(loaded_model_json)

# and weight your nodes with your saved values
model.load_weights('model1.h5')
    
#curl -i -X PUT -F name=Test -F filedata=@SomeFile.pdf "http://localhost:5000/"
@app.route("/")
def predict():
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    #if flask.request.method == "POST":
    tweet = request.args['name']
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