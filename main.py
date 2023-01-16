import nltk
from nltk.stem.lancaster import LancasterStemmer

import specbot

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
#importing modules

tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR) #disabling some flags
with open("intents.json") as file: #opening json file
    data = json.load(file)
debug = False

try:
    with open("data.pickle", "rb") as f: #opening saved data from pickle file
        words, labels, training, output = pickle.load(f)
except: # if no pickle, getting data and putting it into pickle file
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]: #patterns are the things the ai will look for
            wrds = nltk.word_tokenize(pattern) #using nltk to tokenize the words/phrases in the patterns
            words.extend(wrds) #adding to the list
            docs_x.append(wrds) #x = things to look for
            docs_y.append(intent["tag"])# y = things to respond with

        if intent["tag"] not in labels:
            labels.append(intent["tag"]) #attaching all overarching tags to labels file

    words = [stemmer.stem(w.lower()) for w in words if w != "?"] #stems the words to their stems, excluding question marks
    words = sorted(list(set(words)))
    labels = sorted(labels) #sorting the lists

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))] #creates list of zeros

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words: #adds unique stems to bag
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:] #copy of row of zeros
        output_row[labels.index(docs_y[x])] = 1 #puts a one in the position where the stem matches the tag

        training.append(bag)
        output.append(output_row) #adds the values to the main lists

    training = numpy.array(training)
    output = numpy.array(output) #turns lists into arrays

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f) #saves to pickle file, and generates one if one does not exist

tensorflow.compat.v1.reset_default_graph() #resets global default graph to clear data from before

net = tflearn.input_data(shape=[None, len(training[0])]) #input data
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #3 layers of neural network
net = tflearn.regression(net, op_name="stupid-thing")  #turning the data into an algorithm that can be used by regression

model = tflearn.DNN(net) # makes the model to be trained based off the layers

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=2500, batch_size=8, show_metric=True) #training the model
    model.save("model.tflearn")


# model.fit(training, output, n_epoch=1500, batch_size=8, show_metric=True)
# model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)
# this function performs the same duties as what was used before in the pickle file, turning words into their roots
# this is so that they can be predicted with the model - if they match up a 1 is placed where they match

def chat():
    global debug
    print("Welcome to ChatBot! How can I help?")
    responses = []
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            print("Quitting ChatBot...")
            break
        elif inp.lower() == "debug":
            debug=not debug

        results = model.predict([bag_of_words(inp, words)]) #using the input the expected output is predicted
        results_index = numpy.argmax(results)
        x = results[0][results_index]
        tag = labels[results_index]
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses'] #after finding correct response, take the response from the json
        if x > 0.6:  # threshold
            if tag == "specification":
                specbot.chat() #switches to specbot if about the specification
            else:
                print(random.choice(responses))
                print()
        else:
            specbot.errormessage()

        if debug:
            for i in range(len(labels)):
                print(f"{numpy.array_split(results[0], len(labels))[i]*100} chance of {labels[i]}")
            print(f"{x*100} chance of {tag}")


chat() #runs program
