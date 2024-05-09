import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Layer, Dense, Dropout, MultiHeadAttention, LayerNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
from description2genre import load_text, preprocess_text, TransformerEncoder, TokenAndPositionEmbedding, ClassificationModel
import pandas as pd
import pickle

# load data
data = pd.read_csv('../data/data_class.csv')

data_one_hot = pd.get_dummies(data['Genres'], prefix='Genres')
data = pd.concat([data, data_one_hot], axis=1)

sentences = data['About the game']
label = data.loc[:, data.columns != 'About the game']

X_train, X_val, y_train, y_val = train_test_split(sentences, label, test_size=0.2, stratify=label, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, stratify=y_val, random_state=42)
print('>>> train:', X_train.shape, y_train.shape)
print('>>> validation:', X_val.shape, y_val.shape)
print('>>> test:', X_test.shape, y_test.shape)

max_len = 40       
oov_token = '00_V' 
padding_type = 'post'
trunc_type = 'post'  

tokenizer = Tokenizer(char_level=False, lower=True, split=' ')
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1
print("Vocab Size: ",vocab_size)

X_train = X_train.astype(str).tolist()
X_val = X_val.astype(str).tolist()
X_test = X_test.astype(str).tolist()

train_sequences = tokenizer.texts_to_sequences([X_train])
X_train = pad_sequences(train_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

val_sequences = tokenizer.texts_to_sequences([X_val])
X_val = pad_sequences(val_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences([X_test])
X_test = pad_sequences(test_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)

# Save the tokenizer to a file using pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


vocab_size = vocab_size  # Example vocabulary size
num_classes = 6    # Example number of classes
seq_length = 40    # Example sequence length
embed_dim = 256     # Embedding size
heads = 3           # Number of attention heads
hidden_size = 256   # Size of the dense layer
batch_size = 32     # Batch size


def run_classification_model(epochs): 
    model = ClassificationModel(embed_dim, heads, hidden_size, seq_length, vocab_size, num_classes, batch_size)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    x_train = np.random.randint(0, vocab_size, (100, seq_length))
    y_train = np.random.randint(0, num_classes, (100,))

    # train
    count=0
    stats = []
    print('------Training------')
    for epoch in range(epochs):
        count += 1
        train_loss, train_accuracy = model.train_step(X_train, y_train, batch_size)
        stats += [loss, accuracy]
        print(f'>>> epoch {epoch}: loss = {train_loss}, accuracy = {train_accuracy}')
    
    # test
    print('------Testing------')
    test_loss, test_accuracy = model.test_step(X_test, y_test)

    return model


model = run_classification_model(3)
model.summary()
model.save('../model/genre_classification_model.h5')


