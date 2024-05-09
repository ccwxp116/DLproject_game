import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import pickle

# load tokenizer
with open('rnn_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('genre_names.pkl', 'rb') as f:
    genre_names = pickle.load(f)

def sentence_tokenize(text):
    new_sequences = tokenizer.texts_to_sequences(text)
    new_padded_sequences = pad_sequences(new_sequences, maxlen=200, padding='post', truncating='post')
    return new_padded_sequences

def genre_classifier(text):
    predictions = model.predict(text)
    predicted_genres = np.argmax(predictions, axis=1)

    predicted_genre_names = [genre_names[idx] for idx in predicted_genres]
    print(predicted_genre_names)
    return predicted_genres


# Load the model
model_path = '../model/genre_classification_model_rnnsize256_bidirectional.h5'
# model_path = '../model/genre_classification_model_rnnsize256_bidirectional_dropout0.2.h5'
model = load_model(model_path)

text = [
    'when the Roman people honor a warrior and his own strategy and an unforgettable experience in this game you have been designed for a variety of weapons and abilities to explore the dungeons and find out what happened to the world.',
    'discover a thrilling world of adventure with a variety of weapons and abilities as possible to get out of the water and more than than 5 0 0 levels of the game modes and to get ready to fight against the waves of enemies in the game.',
    'when the Roman people honor a warrior by sent to the bottom of the most powerful forces to fight against the undead or protect your home or other players to fight against the enemies or destroy them with your friends',
]

encoded = sentence_tokenize(text)
print(genre_classifier(encoded))