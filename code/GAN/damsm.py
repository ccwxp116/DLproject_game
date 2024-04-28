# Reference: adapted from https://github.com/tobran/DF-GAN
import tensorflow as tf
from tensorflow.keras import layers, models, applications

# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(tf.keras.Model):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = 18
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = 'LSTM'
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.encoder = layers.Embedding(self.ntoken, self.ninput)
        self.drop = layers.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = layers.LSTM(self.nhidden, return_sequences=True,
                                   return_state=True, dropout=self.drop_prob,
                                   recurrent_dropout=self.drop_prob)
            if self.bidirectional:
                self.rnn = layers.Bidirectional(self.rnn)
        elif self.rnn_type == 'GRU':
            self.rnn = layers.GRU(self.nhidden, return_sequences=True,
                                  return_state=True, dropout=self.drop_prob,
                                  recurrent_dropout=self.drop_prob)
            if self.bidirectional:
                self.rnn = layers.Bidirectional(self.rnn)
        else:
            raise NotImplementedError

    def call(self, captions, cap_lens, hidden, mask=None):
        # input: tf.int32 of shape batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        
        # RNN input shape: (batch_size, timesteps, input_dim)
        output, state_h, state_c = self.rnn(emb, mask=mask, initial_state=hidden)
        
        # output shape: (batch_size, timesteps, num_directions * hidden_size)
        # state_h shape: (num_layers * num_directions, batch_size, hidden_size)
        # state_c shape: (num_layers * num_directions, batch_size, hidden_size)
        
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = tf.transpose(output, perm=[0, 2, 1])
        
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = tf.reshape(state_h, shape=[-1, self.nhidden * self.num_directions])
        else:
            sent_emb = tf.reshape(state_h, shape=[-1, self.nhidden * self.num_directions])
        
        return words_emb, sent_emb


def conv1x1(in_planes, out_planes, bias=False):
    """1x1 convolution with padding"""
    return layers.Conv2D(out_planes, kernel_size=1, strides=1,
                         padding='same', use_bias=bias)


class CNN_ENCODER(tf.keras.Model):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        self.nef = 256  # define a uniform ranker

        model = applications.InceptionV3(include_top=False, weights='imagenet')
        
        # Freeze the weights of the pre-trained model
        model.trainable = False
        
        self.Conv2d_1a_3x3 = model.get_layer('conv2d_1a_3x3')
        self.Conv2d_2a_3x3 = model.get_layer('conv2d_2a_3x3')
        self.Conv2d_2b_3x3 = model.get_layer('conv2d_2b_3x3')
        self.Conv2d_3b_1x1 = model.get_layer('conv2d_3b_1x1')
        self.Conv2d_4a_3x3 = model.get_layer('conv2d_4a_3x3')
        self.Mixed_5b = model.get_layer('mixed_5b')
        self.Mixed_5c = model.get_layer('mixed_5c')
        self.Mixed_5d = model.get_layer('mixed_5d')
        self.Mixed_6a = model.get_layer('mixed_6a')
        self.Mixed_6b = model.get_layer('mixed_6b')
        self.Mixed_6c = model.get_layer('mixed_6c')
        self.Mixed_6d = model.get_layer('mixed_6d')
        self.Mixed_6e = model.get_layer('mixed_6e')
        self.Mixed_7a = model.get_layer('mixed_7a')
        self.Mixed_7b = model.get_layer('mixed_7b')
        self.Mixed_7c = model.get_layer('mixed_7c')

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = layers.Dense(self.nef)

    def call(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = tf.image.resize(x, size=(299, 299))
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = layers.AveragePooling2D(pool_size=(8, 8))(x)
        # 1 x 1 x 2048
        x = tf.reshape(x, shape=(x.shape[0], -1))
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code