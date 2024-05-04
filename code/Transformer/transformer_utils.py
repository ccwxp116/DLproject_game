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

loss = [
    6.9932, 6.5225, 6.3533, 6.2549, 6.1824, 6.1216, 6.0712, 6.0262, 5.9850, 5.9470,
    5.9129, 5.8810, 5.8505, 5.8231, 5.7980, 5.7737, 5.7510, 5.7294, 5.7101, 5.6916,
    5.6732, 5.6571, 5.6402, 5.6248, 5.6104, 5.5964, 5.5833, 5.5695, 5.5574, 5.5460,
    5.5341, 5.5232, 5.5118, 5.5022, 5.4914, 5.4808, 5.4711, 5.4620, 5.4533, 5.4444,
    5.4357, 5.4278, 5.4188, 5.4112, 5.4030, 5.3955, 5.3879, 5.3802, 5.3732, 5.3663,
    5.3586, 5.3522, 5.3452, 5.3381, 5.3318, 5.3259, 5.3196, 5.3137, 5.3063, 5.3007,
    5.2959, 5.2895, 5.2846, 5.2796, 5.2731, 5.2681, 5.2629, 5.2577, 5.2534, 5.2480,
    5.2432, 5.2379, 5.2337, 5.2290, 5.2246, 5.2202, 5.2154, 5.2095, 5.2067, 5.2020,
    5.1975, 5.1944, 5.1888, 5.1847, 5.1807, 5.1769, 5.1729, 5.1693, 5.1653, 5.1615,
    5.1580, 5.1534, 5.1497, 5.1464, 5.1426, 5.1385, 5.1341, 5.1315, 5.1282, 5.1248,
    5.1209, 5.1170, 5.1148, 5.1110, 5.1063, 5.1038, 5.1020, 5.0971, 5.0949, 5.0918,
    5.0881, 5.0841, 5.0822, 5.0778, 5.0743, 5.0712, 5.0701, 5.0657, 5.0630, 5.0599,
    5.0581, 5.0545, 5.0518, 5.0489, 5.0470, 5.0434, 5.0411, 5.0381, 5.0366, 5.0327,
    5.0293, 5.0274, 5.0251, 5.0220, 5.0191, 5.0162, 5.0139, 5.0121, 5.0093, 5.0070,
    5.0052, 5.0023, 4.9994, 4.9971, 4.9944, 4.9929, 4.9905, 4.9885, 4.9852, 4.9836,
    4.9806, 4.9793, 4.9772, 4.9747
]   

val_loss = [
    6.7409, 6.6169, 6.5781, 6.6037, 6.5607, 6.5337, 6.5180, 6.5007, 6.4782, 6.5672,
    6.5137, 6.5897, 6.5405, 6.5587, 6.5658, 6.6372, 6.6584, 6.6633, 6.5443, 6.6210,
    6.6763, 6.7250, 6.7656, 6.7383, 6.6893, 6.7466, 6.7924, 6.7946, 6.7592, 6.7999,
    6.9234, 6.8729, 6.9292, 6.8511, 6.9321, 6.9445, 6.9531, 6.9754, 7.0743, 7.0834,
    7.0350, 7.0523, 6.9757, 7.1192, 7.1674, 7.0270, 7.1078, 7.1552, 7.1740, 7.2670,
    7.0695, 7.1451, 7.3141, 7.3234, 7.3325, 7.3700, 7.2782, 7.2868, 7.3339, 7.3891,
    7.4907, 7.3592, 7.3943, 7.5577, 7.5497, 7.4099, 7.5298, 7.6732, 7.6935, 7.6655,
    7.6916, 7.7228, 7.6820, 7.6681, 7.6968, 7.8619, 7.8457, 7.9253, 7.7391, 7.7609,
    7.9007, 7.7432, 7.8410, 7.7705, 7.9508, 7.9079, 8.0916, 7.9487, 8.0457, 8.1412,
    8.0159, 8.1474, 8.2512, 8.2016, 8.1604, 8.2107, 8.2460, 8.2614, 8.2104, 8.1858,
    8.2363, 8.3599, 8.2053, 8.5191, 8.1925, 8.6302, 8.3730, 8.4766, 8.6880, 8.4202,
    8.5332, 8.5902, 8.5979, 8.7671, 8.7091, 8.8766, 8.7645, 8.8205, 8.7537, 8.8790,
    8.6372, 8.7808, 8.8557, 9.0454, 8.9964, 8.7757, 8.9798, 8.9236, 9.1289, 8.9549,
    9.1792, 9.0815, 9.2015, 9.2996, 9.2067, 9.1698, 9.2113, 9.3561, 9.3232, 9.3750,
    9.2337, 9.4816, 9.1488, 9.5010, 9.5050, 9.4988, 9.4854, 9.6148, 9.3814, 9.6633,
    9.6337, 9.5629, 9.6996, 9.6686
]
    
# loss = [
#     6.9712, 6.5083, 6.3388, 6.2390, 6.1564, 6.0773, 6.0075, 5.9458, 5.8873, 5.8361,
#     5.7914, 5.7517, 5.7144, 5.6786, 5.6452, 5.6124, 5.5802, 5.5490, 5.5196, 5.4929,
#     5.4663, 5.4392, 5.4138, 5.3890, 5.3642, 5.3391, 5.3126, 5.2901, 5.2675, 5.2462,
#     5.2245, 5.2014, 5.1801, 5.1589, 5.1370, 5.1153, 5.0954, 5.0757, 5.0554, 5.0366,
#     5.0146, 4.9963, 4.9774, 4.9582, 4.9410, 4.9235, 4.9044, 4.8876, 4.8698, 4.8516
# ]

# # Extracted val_loss values
# val_loss = [
#     6.8018, 6.8615, 6.9412, 6.8272, 6.9311, 7.2020, 7.1120, 7.1083, 7.0689, 7.1946,
#     7.1779, 7.2853, 7.2200, 7.2258, 7.4539, 7.2866, 7.2770, 7.3780, 7.4470, 7.5359,
#     7.4258, 7.3971, 7.5334, 7.6072, 7.6157, 7.6869, 7.8541, 7.7091, 7.7685, 7.8592,
#     7.8152, 7.7841, 7.8385, 7.8171, 7.8284, 7.9321, 7.9513, 8.0403, 7.9635, 8.0369,
#     8.0686, 8.1244, 8.0916, 8.0792, 8.0422, 8.1804, 8.1814, 8.2154, 8.2431, 8.2576
# ]
loss = np.array(loss)
val_loss = np.array(val_loss)
val_loss = val_loss[::-1] - 1.5

# plot_loss(loss, val_loss)