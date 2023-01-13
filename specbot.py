import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
with open("specbot.json") as file:
    data = json.load(file)
debug = False

try:
    with open("data2.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data2.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net,op_name="specbot")

model = tflearn.DNN(net)

try:
    model.load("specmodel.tflearn")
except:
    model.fit(training, output, n_epoch=2500, batch_size=8, show_metric=True)
    model.save("specmodel.tflearn")
# model.fit(training, output, n_epoch=1500, batch_size=8, show_metric=True)
# model.save("specmodel.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    global debug
    print("Welcome to SpecBot! How can I help?")
    print("You can find out more about the specification here: https://www.ocr.org.uk/Images/558027-specification-gcse-computer-science-j277.pdf")
    responses = []
    while True:
        inp = input("You: ")

        if inp.lower() == "quit":
            print("Quitting SpecBot...")
            print("Welcome to ChatBot! How can I help?")
            break
        elif inp.lower() == "debug":
            debug=not debug


        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        x=results[0][results_index]
        tag = labels[results_index]
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        if x>0.5: #threshold
             print(random.choice(responses))
             print()
        else:
            errormessage()
        if debug:
            for i in range(len(labels)):
                print(f"{numpy.array_split(results[0], len(labels))[i]*100} chance of {labels[i]}")
            print(f"{x*100} chance of {tag}")


def errormessage(): #prints an error message
    errormessages = ["I'm not sure I understand. Please make sure everything is spelled correctly and try again.",
                     "I don't understand what you are talking about. Make sure you have spelled everything correclty and try again."]
    print(random.choice(errormessages))
    print('Type "help" if you need more help')



