from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
# import preprocess
import pickle
import numpy as np
import time
import tensorflow as tf
from transformer_utils import LookAheadMaskLayer

###################################################
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
###################################################

# vocab_size = 19154

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Path to the .h5 file
# 这个结果还可以
# model_path = '../model/transformer_batch32_epoch200_embedding64_headsize128_numhead2_dropout0.5_lr1e-4_separatepunc_5k.h5'
# seqlength农场了以后好了些
model_path = '../model/transformer_batch32_epoch50_seqlen60_embedding64_headsize128_numhead3_dropout0.5_ffdim128_lr1e-4_separatepunc_5k.h5'
# seqlength=80
# model_path = '../model/transformer_batch32_epoch50_seqlen80_embedding64_headsize128_numhead3_dropout0.5_ffdim128_lr1e-4_separatepunc_5k.h5'

# model_path = '../model/transformer_batch32_epoch30_seqlen60_2decoder_embedding64_headsize128_numhead3_dropout0.5_ffdim128_lr1e-4_separatepunc_5k.h5'

# Load the model
model = load_model(model_path, 
                   custom_objects={'LookAheadMaskLayer': LookAheadMaskLayer})
# model.summary()

def generate_text(model, tokenizer, start_text, num_generated=103):
    result = start_text.split()
    for _ in range(num_generated):
        encoded = tokenizer.texts_to_sequences([result])[-1]
        encoded = pad_sequences([encoded], maxlen=60, truncating='pre')
        yhat = np.argmax(model.predict(encoded), axis=-1)
        out_word = tokenizer.index_word[yhat[0]]
        result.append(out_word)
    return ' '.join(result)


# Example usage
# text = 'Super Ubie Island REMIX is an adorable platformer'
text = 'discover a thrilling world of adventure'
print(generate_text(model, tokenizer, text))