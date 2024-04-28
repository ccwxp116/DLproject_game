import numpy as np 
import pandas as pd  
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
import time
import pickle
import os
import preprocess

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

# # get the data
train_id, test_id, vocab = preprocess.get_data("../data/nlp_train.txt", "../data/nlp_test.txt")
# train_id = np.array(train_id)
# test_id  = np.array(test_id)
# X0, Y0 = train_id[:-1], train_id[1:]
# X1, Y1  = test_id[:-1],  test_id[1:]
# print(X0.shape, Y0.shape)

# # get vocabulary size
vocab_size = len(vocab)

# def process_data(window_size, data):
#     remainder = (len(data) - 1)%window_size
#     data = data[:-remainder]
#     data = data[:-1].reshape(-1, 20)
#     return data

# X0 = process_data(20, X0)
# Y0 = process_data(20, Y0)
# X1 = process_data(20, X1)
# Y1 = process_data(20, Y1)
# print(X0.shape, Y0.shape)

# read in the data
train_path = "../data/nlp_train.txt"
with open(train_path, 'r', encoding='utf-8') as file:
    data = file.read()

# separate punctuation
def separate_punc(doc_text):
    return [token.lower() for token in doc_text.split(" ") if token not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']

# Clean the text data by removing punctuation and converting tokens to lowercase
data = separate_punc(data)

# Join the cleaned tokens back into a single string
clean_data = " ".join(data)

# Define Tokenizer
tokenizer = Tokenizer(num_words=None, char_level=False)

# Fit the tokenizer on the cleaned text data
tokenizer.fit_on_texts([clean_data])

# Retrieve the word indices from the tokenizer
tokenizer.word_index

# Initialize an empty list to store input sequences
input_sequences = []

# Iterate over each sentence in the cleaned data
for sentence in clean_data.split('\n'):
    # Tokenize the sentence
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
    
    # Iterate over the tokenized sentence to create input sequences
    for i in range(1, len(tokenized_sentence)):
        # Append the input sequence to the list
        input_sequences.append(tokenized_sentence[:i+1])

# Print the first few input sequences for demonstration
print(input_sequences[:20])

# Calculate the maximum sequence length among all input sequences
max_len = max([len(x) for x in input_sequences])
print('>>> max len of input_sequences:', max_len)

# Pad the input sequences to ensure uniform length
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

# Extract features (X) and labels (y)
X = padded_input_sequences[:, :-1]
y = padded_input_sequences[:, -1]

# Print the shapes of X and y
print('>>> X.shape:', X.shape)
print('>>> y.shape:', y.shape)

# Perform one-hot encoding on the labels (y)
y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)
y_onehot_shape = y.shape[1]
print('>>> after one-hot encode on y:', y.shape)

# Define the LSTM language model architecture
model = Sequential([
    Embedding(y_onehot_shape, 10, input_length=(max_len-1)),
    LSTM(15),
    Dense(y_onehot_shape, activation="softmax"),
])

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Print the model summary
model.summary()

# Train the LSTM language model
model.fit(X, y, epochs=50)

#save model
model.save("10_15_epoch50.h5")
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)