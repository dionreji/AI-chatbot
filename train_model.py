import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Libraries needed for Tensorflow processing
import tensorflow as tf
import numpy as np
import tflearn
import random
import json

file_path = 'intents.json'

# Open and read the JSON file
with open(file_path, 'r') as file:
    intents = json.load(file)

words = []          # List of all words -- we create one hot encoding for each word
classes = []        # List of all classes/labels deciding output -- same as tags in intents.json
documents = []      # List of all sentences(patterns in intents.json) and their corresponding tags

for index in intents['intents']:
    for pattern in index['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, index['tag']))
        if index['tag'] not in classes:
            classes.append(index['tag'])

stemmed_words = sorted(set([stemmer.stem(word.lower()) for word in words if word not in ['?', '!', '.']]))

print('Words:', stemmed_words)

input_data = []
output_data = []
for doc in documents:
    sentence = doc[0]
    tag = doc[1]
    sentence = [stemmer.stem(word.lower()) for word in sentence]
    input_data.append([1 if word in sentence else 0 for word in stemmed_words])
    out = [1 if tag == c else 0 for c in classes]
    output_data.append(out)

input_data = np.array(input_data)
output_data = np.array(output_data)

# Neural Network Model
def create_model(input_data, output_data):
    tf.compat.v1.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(input_data[0])])
    net = tflearn.fully_connected(net, 10)
    net = tflearn.fully_connected(net, 10)
    net = tflearn.fully_connected(net, len(output_data[0]), activation='softmax')
    net = tflearn.regression(net)
    model = tflearn.DNN(net)
    return model

model = create_model(input_data, output_data)
model.fit(input_data, output_data, n_epoch=1000, batch_size=8, show_metric=False)
model.save('model.tflearn')

print("Model trained and saved to 'model.tflearn'")

with open('words.json', 'w') as f:
    json.dump(stemmed_words, f)

with open('classes.json', 'w') as f:
    json.dump(classes, f)
print("Words and classes saved to resp. json files.")

