"""Deploying imagenet sample model"""

import os
import io
import flask
import json
import pickle
import keras
import keras.preprocessing.text as kpt
from keras.models import model_from_json
import numpy as np
from keras.preprocessing.text import Tokenizer
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
app = flask.Flask(__name__)
CORS(app)
tokenizer = Tokenizer(num_words=3000)

# for human-friendly printing
labels = ['negative', 'neutral','positive']
with open('/tmp/dictionary.json', 'r') as dictionary_file:
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

vectorizer = pickle.load(open('/tmp1/vectorizer.sav', 'rb'))
classifier = pickle.load(open('/tmp1/sgdclassifier.sav', 'rb'))

#curl -i -X PUT -F name=Test -F filedata=@SomeFile.pdf "http://localhost:5000/"
@app.route("/")
def predict():
    # data = {"success": False}
    model_number=request.args['model_number']
    print("model number is " + model_number)
    tweet = request.args['name']
    if model_number=="0":
        # JSON FOR SNN

        json_file = open('/tmp/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        # and create a model from that
        model = model_from_json(loaded_model_json)
        model.load_weights('/tmp/model1.h5')
        
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

    if model_number=="1":
        data = {}
        text_vector = vectorizer.transform([tweet])
        result = classifier.predict(text_vector)
        data["prediction"] = (labels[result[0]])
        return flask.jsonify(data)


if __name__ == '__main__':
    print("Loading model and starting Flask Server..")
    #load_model()
    app.run(host='0.0.0.0', port=8001, threaded=False)