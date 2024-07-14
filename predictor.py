import tensorflow as tf
import numpy as np
import json
import random
from train_model import model
from datetime import datetime

import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Load the saved model
model.load('./model.tflearn')

# Display the model's architecture
# model.summary()

# Load the list of words and classes from the JSON files
with open('words.json', 'r') as f:
    words = json.load(f)
with open('classes.json', 'r') as f:
    classes = json.load(f)
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Create a function to preprocess the input sentence
def preprocess(sentence):
    sentence = nltk.word_tokenize(sentence)
    sentence = [stemmer.stem(word.lower()) for word in sentence]
    return sentence

# create input vector
def create_input(sentence):
    input_data = [0] * len(words)
    sentence = preprocess(sentence)
    for word in sentence:
        for i, w in enumerate(words):
            if w == word:
                input_data[i] = 1
    return np.array(input_data).reshape(1, -1)

# Create a function to predict the class of the input sentence
def predict_class(sentence):
    input_data = create_input(sentence)
    results = model.predict(input_data)
    results_index = np.argmax(results)
    tag = classes[results_index]
    return tag

# Create a function to get the response
def get_response(tag):
    if tag == "datetime":
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if tag == "goodbye":
        return "Goodbye! Have a nice day! 🤖"   
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Create a function to chat with the bot
def chat():
    print("Start talking with the bot (type 'quit' to stop)!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        tag = predict_class(user_input)
        response = get_response(tag)
        print("Bot:", response)

# Start chatting with the bot
# chat()
