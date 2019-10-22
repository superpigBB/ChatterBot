import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.stem.lancaster import LancasterStemmer

import json
with open('intents.json', 'r') as file:
    data = json.load(file)

sentences = []
labels = []
total_words = []
stemmer = LancasterStemmer()

for item in data['intents']:
    for sentence in item['patterns']:
        words = nltk.word_tokenize(sentence)
        words = [stemmer.stem(w.lower()) for w in words if w != '?']
        total_words.extend(words)
        sentences.append(words)

    labels.append(item['tag'])

print(f"patterns: {sentences} \ntags: {labels}")

max_phrase_len = max([len(phrase) for phrase in sentences])
print(f"max phrase len of all: {max_phrase_len}")

total_word = len(set(total_words))
print(f"total distinct words length: {total_word}")

tokenizer = Tokenizer(oov_token = "<OOV>")  # total_word should be all unique values across words
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)  # word vector length is not the same

print(f"sentences: {sentences}")
print(f"word_index: {word_index}")
print(f"sequences: {sequences}")

# uniform tokenized senteces using padding
padded = pad_sequences(sequences, padding='post')
print(f"padded sequences: \n {padded}")  # make all sentence the same length
print(f"padded[0]: {padded[0]}")
print(f"padded_shape: {padded.shape}")