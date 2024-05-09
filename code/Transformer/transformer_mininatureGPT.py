import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
# from keras import ops
from keras.layers import TextVectorization
import numpy as np
import os
import string
import random
import tensorflow as tf
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout, GlobalAveragePooling1D

###################################################################
# TRANSFORMER DECODER
###################################################################
class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, num_heads, dropout, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.num_heads = num_heads
        self.dropout = dropout

        # embedding
        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.hidden_size)
        self.encoding = PositionalEncoding(vocab_size, hidden_size)
        self.decoder = TransformerBlock(self.hidden_size, self.num_heads, self.dropout)
        self.final_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs, training=False):
        # Positional encoding
        x = self.pos_encoding(inputs)

        # Decoder blocks
        x = self.decoder(x)

        if training:
            x = tf.keras.layers.Dropout(rate=0.4)

        # Final dense layer to predict the next word
        logits = self.final_layer(x)

        return logits
    
    def get_config(self):
        config = super(TransformerDecoder, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'window_size': self.window_size,
            'num_heads' : self.num_heads,
        })
        return config

###################################################################
# TRANSFORMER BLOCK
###################################################################
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, emb_sz, num_head, dropout, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)

        self.ff_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(emb_sz, activation='relu', kernel_regularizer=l2(0.01)),  
            tf.keras.layers.Dense(emb_sz, kernel_regularizer=l2(0.01))  
        ])
        self.emb_sz = emb_sz
        self.num_head = num_head
        self.dropout = dropout

        self.self_atten =  MultiHeadAttention(key_dim=self.emb_sz, num_heads=self.num_head, dropout=self.dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    @tf.function
    def call(self, inputs):
        """
        This functions calls a transformer block.
        :param inputs: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ] encoded output text
        :return: tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE ]
        """
        # print('------In transformerBlock------')
        # print('>>> self.emb)sz:', self.emb_sz)
        # print('>>> input.shape[BATCH_SIZE x INPUT_SEQ_LENGTH x EMBEDDING_SIZE]:', inputs.shape)

        # 1) normalize
        normalized_inputs = self.layer_norm(inputs)  
        # print('>>> normalized_self_attn.shape:', normalized_self_attn.shape)

        # 2) self attention
        batch_size = normalized_inputs[0]
        seq_len = normalized_inputs[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, "bool")
        attn_output = self.self_atten(normalized_inputs, normalized_inputs, normalized_inputs, attention_mask=causal_mask)
        # print('>>> context_attn_output.shape:', context_attn_output.shape)

        # 3) add & normalize
        normalized_attn = self.layer_norm(normalized_inputs+attn_output)
        # print('>>> normalized_context_attn.shape:', normalized_context_attn.shape)  

        # 4) feed forward layer
        ff_output = self.ff_layer(normalized_attn)
        # print('>>> ff_output.shape:', ff_output.shape)

        # 5) add & normalize
        normalized_ff = self.layer_norm(normalized_attn + ff_output)  
        # print('>>> normalized_ff.shape:', normalized_ff)

        # 6) return relu of tensor
        return tf.nn.relu(normalized_ff)
    
###################################################################
# POSITIONAL ENCODING
###################################################################
def positional_encoding(length, depth):
    depth = depth/2
    ## Generate a range of positions and depths 
    positions = np.arange(length)[:, np.newaxis]    # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth  # (1, depth)
    ## Compute range of radians to take the sine and cosine of.
    angle_rates = 1 / (10000**depths)               # (1, depth)
    angle_rads = positions * angle_rates            # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    ## This serves as offset for the Positional Encoding
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=self.embed_size)
        self.pos_encoding = positional_encoding(length=window_size, depth=self.embed_size)

    def call(self, x):
        x = self.embedding(x) 
        x *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
        x += self.pos_encoding[:tf.shape(x)[1],:]
        return x
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'embed_size': self.embed_size,
        })
        return config
    
###################################################################
# MASKING
###################################################################
def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    # Use tf.range for creating ranges and expand dimensions where necessary
    i = tf.range(n_dest)[:, tf.newaxis]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat([
        tf.expand_dims([batch_size], -1), 
        tf.constant([[1, 1]], dtype=tf.int32)
    ], axis=0)

    return tf.tile(mask, mult)

    
###################################################################
# TEXT GENERATOR
###################################################################
class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(
        self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1
    ):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        preds = tf.nn.softmax(tf.expand_dims(logits, 0))[0]
        preds = preds.numpy().astype("float32")
        indices = indices.numpy().astype("int32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x, verbose=0)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        print(f"generated text:\n{txt}\n")
