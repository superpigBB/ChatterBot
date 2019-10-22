import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

oov_tok = "<OOV>"
tokenizer = Tokenizer()

input_data= "hELLO! \n good morning\n good bye!\n what's your name? \n what's your age?"
corpus = input_data.lower().split('\n')
print(f"corpus: {corpus}")

tokenizer.fit_on_texts(corpus)
# for key in tokenizer.word_index:
#     tokenizer.word_index[stemmer.stem(key)] = tokenizer.word_index.pop(key)

# total_words = len(tokenizer.word_index) + 1
# print(f"word_index:{tokenizer.word_index}")
# print(f"total words: {total_words}")

label_data = "greeting\n greeting \n bye \n general \n general \n"
label_corpus = label_data.lower().split('\n')
print(f"label corpus: {label_corpus}")

# tokenizer_label = Tokenizer()
tokenizer.fit_on_texts(label_corpus)
total_label_words = len(tokenizer.word_index) + 1
# print(f"label_word_index:{tokenizer.word_index}")
# print(f"total label words: {total_label_words}")

total_words = len(tokenizer.word_index) + 1
print(f"word_index:{tokenizer.word_index}")
print(f"total words: {total_words}")

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    input_sequences.append(token_list)

max_sequence_len = max(len(x) for x in input_sequences)

print(f"input_sequences:{input_sequences}\nmax_sequence_len: {max_sequence_len}")

# input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='post'))
print(f"padded input_sequences: \n{input_sequences}")  # shape: 3 * 5

label_sequences = []
for line in label_corpus:
    if line == '':
        continue
    token_list = tokenizer.texts_to_sequences([line])[0]
    label_sequences.append(token_list)

label_max_sequence_len = max([len(x) for x in label_sequences])
print(f"label_sequences:{label_sequences}\nlabel_max_sequence_len: {label_max_sequence_len}")

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='post'))
label_sequences = np.array(pad_sequences(label_sequences, maxlen=label_max_sequence_len, padding='post'))

xs = input_sequences
print(f"xs: \n{input_sequences}\nlabels: \n{label_sequences}")

ys = tf.keras.utils.to_categorical(label_sequences, num_classes=total_label_words)
print(f"ys: \n{ys }")

model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len))
# model.add(Bidirectional(LSTM(20)))  # LSTM(150)
model.add(LSTM(20))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(lr=0.01)  # learning rate
model.compile(loss='categorical_crossentropy', optimizer=adam,  metrics=['accuracy'])
history = model.fit(xs, ys, epochs=500, verbose=1)

import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

plot_graphs(history, 'accuracy')

# prediction
seed_text = "What can I call your name？"
token_list = tokenizer.texts_to_sequences([seed_text])[0]
token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='post')
predicted = model.predict_classes(token_list, verbose=0)
prob = model.predict_proba(token_list)
print(f"predicted: {predicted}")
print(f"prob: {prob * 100}")
print(f"max index of prob is {np.argmax(prob)}")

dict = dict([(value, key) for (key, value) in tokenizer.word_index.items()])
print(f"tag: {dict[predicted[0]]}")



# e = model.layers[0]
# weights = e.get_weights()[0]
# print(weights.shape)  # shape: (vocab_size, embedding_dim)
# (vocab_size, embedding_dim) = weights.shape
#
# import io
#
# out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
# out_m = io.open('meta.tsv', 'w', encoding='utf-8')
# for word_num in range(1, vocab_size):
#     word = dict[word_num]
#     embeddings = weights[word_num]
#     out_m.write(word + "\n")
#     out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')
# out_v.close()
# out_m.close()
#
# try:
#     from google.colab import files
# except ImportError:
#     pass
# else:
#     files.download('vecs.tsv')
#     files.download('meta.tsv')

pass