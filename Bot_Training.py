"""
AI Neural Network Training For Captain Pi
Created By: Vicky Bao
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, GRU
from tensorflow.keras.models import Sequential, load_model, model_from_json
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

query = 'SELECT * FROM training_bot'

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

# print(f"input_corpose:{input_corpus} \nlabel_corpose: {label_corpus}")

connection.close()

# Tokenization on texts
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words=150, oov_token=oov_tok)

tokenizer.fit_on_texts(input_corpus)
# print(f"input word index: {tokenizer.word_index}")
tokenizer.fit_on_texts(label_corpus)
total_words = len(tokenizer.word_index) + 1
print(f"total words: {tokenizer.word_index}"
      f"total word index: {tokenizer.word_index}")


# Sequence all texts
input_sequences = tokenizer.texts_to_sequences(input_corpus)

# max sequence length of the input
max_sequence_len = max(len(x) for x in input_sequences)

# print(f"input_sequences:{input_sequences}\nmax_sequence_len: {max_sequence_len}")

label_sequences = tokenizer.texts_to_sequences(label_corpus)

label_max_sequence_len = max([len(x) for x in label_sequences])
# print(f"label_sequences:{label_sequences}\nlabel_max_sequence_len: {label_max_sequence_len}")

# Padded sequences
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='post'))
label_sequences = np.array(pad_sequences(label_sequences, maxlen=label_max_sequence_len, padding='post'))

# x and y set up
xs = input_sequences
# print(f"xs: \n{input_sequences}\nlabels: \n{label_sequences}")

# Categorize y before training the model
ys = tf.keras.utils.to_categorical(label_sequences, num_classes=total_words)
# print(f"ys: \n{ys }")

# model = Sequential()
# model.add(Embedding(total_words, 20, input_length=max_sequence_len))   #64
# # model.add(Bidirectional(LSTM(20)))  # LSTM(150) # GRU(32)
# # model.add(Dense(total_words, activation='relu'))
# model.add(LSTM(20))   #total_words
# model.add(Dense(total_words, activation='softmax'))
# # adam = Adam(lr=0.01)  # learning rate
# adam = Adam()
# model.compile(loss='categorical_crossentropy', optimizer=adam,  metrics=['accuracy'])
# history = model.fit(xs, ys, epochs=500, verbose=1)
# model.summary()

seed = 7
from sklearn.model_selection import train_test_split

model_loaded = True

if model_loaded:
    model = load_model('model_new.h5')
    model.summary()
else:
    # Split into Validation and Training Sets
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.33, random_state=seed)
    model = Sequential()
    model.add(Embedding(total_words, 20, input_length=max_sequence_len))   #64 #20
    # model.add(Bidirectional(LSTM(20)))  # LSTM(150) # GRU(32)
    # model.add(Dense(total_words, activation='relu'))
    model.add(LSTM(20))   #total_words #20
    model.add(Dense(total_words, activation='softmax'))
    adam = Adam(lr=0.01)  # learning rate
    # adam = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=adam,  metrics=['accuracy'])
    model_json = model.to_json()

    # model = load_model('my_model_weights.h5')
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=500, verbose=1)

    model.save("model_new.h5")
    print("Saved model to disk")


# evaluate the model
scores = model.evaluate(xs, ys, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# # Plot Accuracy and loss
# import matplotlib.pyplot as plt
#
# def plot_graphs(history, string):
#     plt.plot(history.history[string])
#     plt.plot(history.history['val_'+string])
#     plt.xlabel('Epochs')
#     plt.ylabel(string)
#     plt.legend([string, 'val_'+string])
#     plt.show()
#
# plot_graphs(history, 'accuracy')
# plot_graphs(history, "loss")

# Reverse key pair
dict = dict([(value, key) for (key, value) in tokenizer.word_index.items()])


# prediction
def predict_test(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='post')
    predicted = model.predict_classes(token_list, verbose=0)
    prob = model.predict_proba(token_list)
    # print(f"predicted: {predicted}")
    # print(f"prob: {prob}")
    max_index = np.argmax(prob)
    top_3_index = (-prob).argsort()[0][:3] #[-3:][::-1]
    print(f"max index of prob is {np.argmax(prob)}")
    print(f"top 3 index of prob is {top_3_index}")
    for i in top_3_index:
        print(f"top3 index is {i} -> {prob[0][i] * 100}%")
    max_prob = prob[0][max_index] * 100
    # print(f"word index: \n{tokenizer.word_index}")
    tags = [dict[i] for i in top_3_index]
    # tag = dict[predicted[0]]
    responses = []  # responses to be returned
    for tag in tags:
        if tag in ['greeting', 'age', 'goodbye']:
            responses.append(''.join(response_dict[tag]))
        else:
            tag_response(seed_text, tag, responses)
            # if tag_response(seed_text, tag, responses):
            #     continue
            #     responses = tag_response(seed_text, tag, responses)
    # print(f"tags: {tags}")
    # print(f"max prob for tag {tag}: {max_prob}%")
    # responses = response_dict[tag]
    print(f"responses: {responses}")
    return tags, responses

import re
def tag_response(seed_text, tag, responses):
    responses_set = response_dict[tag]
    for response in responses_set:
        # print(f"response: {response}")
        if re.search('DUDL', seed_text, flags=re.I) and re.search('DUDL', response, flags=re.I):
            responses.append(response)
            return responses
        elif re.search('Install', seed_text, flags=re.I) and re.search('Install', response, flags=re.I):
            responses.append(response)
            return responses
        elif re.search('escape analysis|FEAP', seed_text, flags=re.I) and re.search('escape analysis|FEAP', response, flags=re.I):
            responses.append(response)
            return responses
        elif re.search('Jira|Pre-GA Jira', seed_text, flags=re.I) and re.search('Jira|Pre-GA Jira', response, flags=re.I):
            responses.append(response)
            return responses
        elif re.search('ARTS', seed_text, flags=re.I) and re.search('ARTS', response, flags=re.I):
            responses.append(response)
            return responses
        elif re.search('Customer ARs|external ARs|ars|remedy', seed_text, flags=re.I) and re.search('Customer ARs|external ARs|ars|remedy', response, flags=re.I):
            responses.append(response)
            return responses
        elif re.search('safelaunch', seed_text, flags=re.I) and re.search('safelaunch', response, flags=re.I):
            responses.append(response)
            return responses
        elif re.search('Trident', seed_text, flags=re.I) and re.search('Trident', response, flags=re.I):
            responses.append(response)
            return responses
        elif re.search('powermax', seed_text, flags=re.I) and re.search('powermax', response, flags=re.I):
            responses.append(response)
            return responses
        elif re.search('DU', seed_text, flags=re.I) and re.search('DU', response, flags=re.I):
            responses.append(response)
            return responses
        elif re.search('DL', seed_text, flags=re.I) and re.search('DL', response, flags=re.I):
            responses.append(response)
            return responses
        elif re.search('SR', seed_text, flags=re.I) and re.search('SR', response, flags=re.I):
            responses.append(response)
            return responses
    return responses.append(str(tag) + " what?")





# predict_test("I want to download DUDL Data")
# predict_test("Show me dudl data")
# predict_test("I want DUDL reports")
# predict_test("definition of DUDL")
# predict_test("Hi Captain Pi")
# predict_test("Bye Captain Pi")
# predict_test("I want to download data")
# predict_test("I want a dashboard")
# predict_test("hi")
# predict_test("Hi")
predict_test("Download DUDL")



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


