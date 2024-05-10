import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import pickle
import os
import tensorflow as tf
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# devices = tf.config.list_physical_devices()
# print("\nDevices: ", devices)

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   details = tf.config.experimental.get_device_details(gpus[0])
#   print("GPU details: ", details)

# from tensorflow.python.framework.config import set_memory_growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)

################################################################
# preprocessing
def load_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def preprocess_text(text, seq_length):
    tokenizer = Tokenizer(char_level=False, lower=True, split=' ')
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])[0]

    vocab_size = len(tokenizer.word_index) + 1
    sequences = [sequences[i:i+seq_length+1] for i in range(len(sequences) - seq_length)]
    
    X, y = zip(*[(seq[:-1], seq[-1]) for seq in sequences])
    X = np.array(X)
    y = to_categorical(y, num_classes=vocab_size)
    
    return X, y, tokenizer, vocab_size




#####################################################################
# other utils
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return tf.convert_to_tensor(mask, dtype=tf.float32)  # (size, size)

def get_positional_encoding(seq_length, embed_dim):
    pos = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
    i = tf.range(embed_dim, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(embed_dim, tf.float32))

    # Compute the positional encodings
    angle_rads = pos * angle_rates

    # Apply sin to even indices in the array; 2i
    sines = tf.math.sin(angle_rads[:, 0::2])
    # Apply cos to odd indices in the array; 2i+1
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]

    return pos_encoding

def loss_function(prbs, labels):
    scce = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs, from_logits=True)
    loss = tf.reduce_sum(scce)
    return loss

def accuracy_function(prbs, labels):
    correct_classes = tf.argmax(prbs, axis=-1) == labels
    accuracy = tf.reduce_mean(tf.cast(correct_classes, tf.float32))
    return accuracy

class LookAheadMaskLayer(tf.keras.layers.Layer):
    def __init__(self, seq_length, **kwargs):
        super().__init__(**kwargs)
        self.seq_length = seq_length

        self.mask = 1 - tf.linalg.band_part(tf.ones((self.seq_length, self.seq_length)), -1, 0)
        # self.mask = tf.expand_dims(self.mask, axis=0)

    def call(self, inputs):
        return self.mask

    def get_config(self):
        config = super().get_config()
        config.update({"seq_length": self.seq_length})
        return config

#####################################################################
# fitting the model
def transformer_decoder(inputs, head_size, num_heads, ff_dim, seq_length, dropout):
    mask = LookAheadMaskLayer(seq_length)(inputs)
    # print('>>> mask:', mask.shape)
    # print('>>> Mask Type:', type(mask))
    # normalize
    # x = LayerNormalization(epsilon=1e-6)(inputs)

    # self attention
    attn_output = MultiHeadAttention(
        key_dim=head_size, 
        num_heads=num_heads,
        # dropout=dropout
        )(inputs, inputs, inputs, attention_mask=mask)
    attn_output = Dropout(dropout)(attn_output)

    # add & normalize
    attn_norm = LayerNormalization(epsilon=1e-6)(attn_output + inputs)
    
    # feed forward
    ff_output = Dense(ff_dim, activation="relu", kernel_regularizer=l2(0.01))(attn_norm)
    ff_output = Dense(inputs.shape[-1], kernel_regularizer=l2(0.01))(ff_output)
    ff_output = Dropout(dropout)(ff_output)

    # add & normalize
    ff_norm = LayerNormalization(epsilon=1e-6)(ff_output + attn_norm) 

    # return linear 
    return ff_norm
    # return tf.nn.relu(ff_norm)

def create_transformer_model(vocab_size, seq_length, embed_dim, head_size, num_heads, ff_dim, dropout):
    # input sequence
    inputs = Input(shape=(seq_length,))

    # token embedding
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)

    # positional encoding
    pos_encoding = get_positional_encoding(seq_length, embed_dim)
    x = x + pos_encoding
    
    # decoder
    x = transformer_decoder(x, head_size, num_heads, ff_dim, seq_length, dropout)
    x = transformer_decoder(x, head_size, num_heads, ff_dim, seq_length, dropout)
    # x = transformer_decoder(x, head_size, num_heads, ff_dim, seq_length, dropout)
    x = GlobalAveragePooling1D()(x)
    # linear
    # x = Dense(vocab_size)(x)

    # softmax
    outputs = Dense(vocab_size, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), 
                  loss='categorical_crossentropy',
                #   loss = loss_fn, 
                  metrics=['accuracy'],
                #   metrics = [accuracy_function]
                  )
    return model

######################################################

# load data
text = load_text('small_combine_tinystory.txt')
X, y, tokenizer, vocab_size = preprocess_text(text, seq_length=60)
print('>>> vocab_size:', vocab_size)

# Save the tokenizer to a file using pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# split train and validation
X0, X1, Y0, Y1 = train_test_split(X, y, test_size=0.20, random_state=42)
print('>>> train:', X0.shape, Y0.shape)
print('>>> validation:', X1.shape, Y1.shape)

model = create_transformer_model(vocab_size, 
                                 X0.shape[1], 
                                 embed_dim=64, 
                                 head_size=128, 
                                 num_heads=3, 
                                 ff_dim=128, 
                                 dropout=0.4)
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
model.fit(X0, Y0, batch_size=32, epochs=5, validation_data=(X1, Y1))#, callbacks=[early_stopping])

model.save('transformer_batch32_epoch5_seqlen60_2decoder_embedding64_headsize128_numhead3_dropout0.4_ffdim128_lr1e-4_tinystory.h5')



#####################################################
# generation
def generate_text(model, tokenizer, start_text, num_generated=68):
    result = start_text.split()
    for _ in range(num_generated):
        encoded = tokenizer.texts_to_sequences([result])[-1]
        encoded = pad_sequences([encoded], maxlen=X0.shape[1], truncating='pre')
        yhat = np.argmax(model.predict(encoded), axis=-1)
        out_word = tokenizer.index_word[yhat[0]]
        result.append(out_word)
    return ' '.join(result)


# text generation
print(generate_text(model, tokenizer, 'when the Roman people honor a simple warrior'))
print(generate_text(model, tokenizer, 'discover a thrilling world of adventure'))



