import json
import keras
import keras.preprocessing.text as kpt
from keras.models import model_from_json
import numpy as np
from keras.preprocessing.text import Tokenizer

# we're still going to use a Tokenizer here, but we don't need to fit it
# tokenizer = Tokenizer(num_words=3000)
tokenizer = Tokenizer(num_words=3000)

# for human-friendly printing
labels = ['negative', 'neutral','positive']


# read in our saved dictionary
with open('/tmp/dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

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

# read in your saved model structure
json_file = open('/tmp/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# and create a model from that
model = model_from_json(loaded_model_json)

# and weight your nodes with your saved values
model.load_weights('/tmp/model1.h5')

while 1:
    #for python 2
    # evalSentence = raw_input('Input a sentence to be evaluated, or Enter to quit: ')
    evalSentence = ""
    evalSentence = input('Input a sentence to be evaluated, or Enter to quit: ')
    if len(evalSentence) == 0:
        break

    print("tweet to test: " + evalSentence)
    testArr = convert_text_to_index_array(evalSentence)
    
    input_value = tokenizer.sequences_to_matrix([testArr], mode='binary')

    pred = model.predict(input_value)
    print("%s sentiment; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
