import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

tokenizer = Tokenizer()

data = "In the twon of Athy one. Leremgt lanigan \n Battered away......."
corpus = data.lower().split('\n')
print(f"corpus: {corpus}")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print(f"word_index:{tokenizer.word_index}")
print(f"total words: {total_words}")

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
print(f"max length of the sequence is: {max_sequence_len}")

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]
print(f"xs: \n{xs}\nlabels: \n{labels}")

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len - 1))
model.add(Bidirectional(LSTM(20)))  # LSTM(150)
# model.add(LSTM(20))
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


# Prediction
seed_text = "Laurence went to dublin"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ''

    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word

print(seed_text)
pass