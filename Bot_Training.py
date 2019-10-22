"""
AI Neural Network Training For Captain Pi
Created By: Vicky Bao
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer

# Query tags related data from mysql
import sqlalchemy
engine = sqlalchemy.create_engine(
                    'mysql+mysqlconnector://root:root@localhost:8889/test',
                     echo=True)

connection = engine.connect()
trans = connection.begin()

query = 'SELECT * FROM training_table'

connection = engine.connect()
results = connection.execute(query)

# Input data initialization
input_corpus, label_corpus = [], []
from collections import defaultdict
response_dict = defaultdict(set)
for row in results:
    # pre-process tag and pattern to make it all lower cases and stemmed to its root word
    tag, pattern, response = row[1].lower(), row[2].lower(), row[3]
    input_corpus.append(pattern)
    label_corpus.append(tag)
    response_dict[tag].add(response)

print(f"input_corpose:{input_corpus} \nlabel_corpose: {label_corpus}")

connection.close()

# Tokenization on texts
oov_tok = "<OOV>"
tokenizer = Tokenizer()

tokenizer.fit_on_texts(input_corpus)
print(f"input word index: {tokenizer.word_index}")
tokenizer.fit_on_texts(label_corpus)
total_words = len(tokenizer.word_index) + 1
print(f"total words: {tokenizer.word_index}"
      f"total word index: {tokenizer.word_index}")


# Sequence all texts
input_sequences = []        # Input
for line in input_corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    input_sequences.append(token_list)

# max sequence length of the input
max_sequence_len = max(len(x) for x in input_sequences)

print(f"input_sequences:{input_sequences}\nmax_sequence_len: {max_sequence_len}")


label_sequences = []    # Label
for line in label_corpus:
    if line == '':
        continue
    token_list = tokenizer.texts_to_sequences([line])[0]
    label_sequences.append(token_list)

label_max_sequence_len = max([len(x) for x in label_sequences])
print(f"label_sequences:{label_sequences}\nlabel_max_sequence_len: {label_max_sequence_len}")

# Padded sequences
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='post'))
label_sequences = np.array(pad_sequences(label_sequences, maxlen=label_max_sequence_len, padding='post'))

# x and y set up
xs = input_sequences
print(f"xs: \n{input_sequences}\nlabels: \n{label_sequences}")

# Categorize y before training the model
ys = tf.keras.utils.to_categorical(label_sequences, num_classes=total_words)
print(f"ys: \n{ys }")

model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len))
# model.add(Bidirectional(LSTM(20)))  # LSTM(150)
model.add(LSTM(20))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(lr=0.01)  # learning rate
model.compile(loss='categorical_crossentropy', optimizer=adam,  metrics=['accuracy'])
history = model.fit(xs, ys, epochs=500, verbose=1)

# Plot Accuracy and loss
import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['loss'])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'loss'])
    plt.show()

plot_graphs(history, 'accuracy')

dict = dict([(value, key) for (key, value) in tokenizer.word_index.items()])
# prediction
def predict_test(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='post')
    predicted = model.predict_classes(token_list, verbose=0)
    prob = model.predict_proba(token_list)
    print(f"predicted: {predicted}")
    print(f"prob: {prob}")
    max_index = np.argmax(prob)
    print(f"max index of prob is {np.argmax(prob)}")
    max_prob = prob[0][max_index] * 100
    print(f"word index: \n{tokenizer.word_index}")

    tag = dict[predicted[0]]
    print(f"tag: {tag}")
    print(f"max prob for tag {tag}: {max_prob}%")
    responses = response_dict[tag]
    print(f"responses: {responses}")

predict_test("I want to download DUDL Data")
predict_test("I want data")



# # Load into Tensorflow model for visualization
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



# # Save the model
# model.SAVE("model.h5")
#
#
# # # Load the model
# # model = keras.models.load_model("model.h5")


