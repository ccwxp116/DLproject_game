from transformer_mininatureGPT import *

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
from keras.layers import TextVectorization
import numpy as np
import os
import string
import random
import tensorflow
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout, GlobalAveragePooling1D
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import pickle

##############################################################
# PREPROCESSING
##############################################################
def load_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def preprocess_text(text, seq_length=50):
    tokenizer = Tokenizer(char_level=False, lower=True, split=' ')
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])[0]

    vocab_size = len(tokenizer.word_index) + 1
    sequences = [sequences[i:i+seq_length+1] for i in range(len(sequences) - seq_length)]
    
    X, y = zip(*[(seq[:-1], seq[-1]) for seq in sequences])
    X = np.array(X)
    y = to_categorical(y, num_classes=vocab_size)
    
    return X, y, tokenizer, vocab_size

#######################################################
# MODEL
#######################################################
vocab_size = 20000  # Only consider the top 20k words
maxlen = 80  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer


def create_model():
    inputs = layers.Input(shape=(maxlen,), dtype="int32")
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam",
        loss=[loss_fn, None],
    )  # No loss and optimization based on word embeddings from transformer block
    return model

#######################################################
# RUN THE MODEL
#######################################################
maxlen = 80  # Max sequence size
embed_dim = 128  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer

# read in the text
text = load_text('../data/nlp_train_4k.txt')
X, y, tokenizer, vocab_size = preprocess_text(text)
print('>>> vocab_size:', vocab_size)

# Save the tokenizer to a file using pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# split train and validation
X0, X1, Y0, Y1 = train_test_split(X, y, test_size=0.20, random_state=42)
print('>>> train:', X0.shape, Y0.shape)
print('>>> validation:', X1.shape, Y1.shape)

# Tokenize starting prompt
word_to_index = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index

start_prompt = "this movie is"
start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
num_tokens_generated = 40
text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, vocab)


model = create_model()
model.summary()

model.fit(X0, Y0, batch_size=32, epochs=200, validation_data=(X1, Y1),callbacks=[text_gen_callback])