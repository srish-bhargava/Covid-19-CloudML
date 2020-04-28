# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import json
import keras
import keras.preprocessing.text as kpt
from keras.models import model_from_json

import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re
#from bs4 import BeautifulSoup
import sys
import os
os.environ['KERAS_BACKEND']='theano'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.callbacks import ModelCheckpoint
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.models import load_model
from numpy import loadtxt


# %%
# we're still going to use a Tokenizer here, but we don't need to fit it
tokenizer = Tokenizer(num_words=3000)


# %%
# for human-friendly printing
labels = ['sports', 'entertainment', 'medical', 'politics']


# %%
# read in our saved dictionary
with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)


# %%
# this utility makes sure that all the words in your input
# are registered in the dictionary
# before trying to turn them into a matrix.
def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            print("'%s' not in training corpus; ignoring." %(word))
    return wordIndices


# %%
# read in your saved model structure
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()


# %%
# and create a model from that
model = model_from_json(loaded_model_json)


# %%
# and weight your nodes with your saved values
model.load_weights('model.h5')


# %%
# okay here's the interactive part
while 1:
    evalSentence = raw_input('Input a sentence to be evaluated, or Enter to quit: ')

    if len(evalSentence) == 0:
        break

    print("tweet to test: " + evalSentence)
    # format your input for the neural net
    testArr = convert_text_to_index_array(evalSentence)
    input = tokenizer.sequences_to_matrix([testArr], mode='binary')
    # predict which bucket your input belongs in
    pred = model.predict(input)
    # and print it for the humons
    print("%s sentiment; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))

