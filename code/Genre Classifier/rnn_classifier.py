import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import pickle

# load data
data = pd.read_csv('../data/data_class.csv')

sentences = data['About the game'].astype(str)  
labels = pd.get_dummies(data['Genres'])  

# save the genre labels
genre_names = labels.columns.tolist()
with open('genre_names.pkl', 'wb') as f:
    pickle.dump(genre_names, f)

# Tokenizing
tokenizer = Tokenizer(char_level=False, lower=True, split=' ')
tokenizer.fit_on_texts(sentences)
vocab_size = len(tokenizer.word_index) + 1
print('>>> vocab size:' ,vocab_size)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')

with open('rnn_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=2470)

def create_model(vocab_size, embedding_dim, rnn_size, num_classes):
    model = Sequential([
    Embedding(vocab_size, embedding_dim, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_size,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_size//2)),
    Dense(rnn_size, activation='relu', kernel_regularizer=l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(learning_rate=1e-4), 
                  metrics=['accuracy'])
    return model

# Model parameters
vocab_size = vocab_size
embedding_dim = 128
rnn_size = 256
num_classes = y_train.shape[1]

# Create the model
model = create_model(vocab_size, embedding_dim, rnn_size, num_classes)
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), batch_size=32, callbacks=[early_stopping])
model.save('../model/genre_classification_model_rnnsize256_bidirectional_dropout0.2.h5')