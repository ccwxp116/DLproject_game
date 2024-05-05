import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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

def generate_text(model, tokenizer, start_text, num_generated=30):
    result = start_text.split()
    for _ in range(num_generated):
        encoded = tokenizer.texts_to_sequences([result])[-1]
        encoded = pad_sequences([encoded], maxlen=50, truncating='pre')
        yhat = np.argmax(model.predict(encoded), axis=-1)
        out_word = tokenizer.index_word[yhat[0]]
        result.append(out_word)
    return ' '.join(result)

def plot_loss(loss, val_loss):
    plt.figure(figsize=(10, 5))  # Optional: Specify the figure size
    plt.plot(np.arange(1, len(loss) + 1), loss, label='Training Loss')
    plt.plot(np.arange(1, len(loss) + 1), val_loss, label='Validation Loss')
    plt.xticks(np.arange(1, len(loss) + 1, step=1))  # Adjust step if too dense
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# plot_loss(loss, val_loss)