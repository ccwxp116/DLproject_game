# Reference: adapted from https://github.com/tobran/DF-GAN
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import Zeros, Ones

class NetG(Model):
    def __init__(self, ngf, nz, cond_dim, imsize, ch_size):
        super(NetG, self).__init__()
        self.ngf = ngf
        self.fc = layers.Dense(ngf*8*4*4)
        self.GBlocks = []
        in_out_pairs = get_G_in_out_chs(ngf, imsize)
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.GBlocks.append(G_Block(cond_dim+nz, in_ch, out_ch, upsample=True))
        self.to_rgb = tf.keras.Sequential([
            layers.LeakyReLU(0.2),
            layers.Conv2D(ch_size, 3, 1, 'same'),
            layers.Activation('tanh')
        ])

    def call(self, noise, c):
        out = self.fc(noise)
        out = tf.reshape(out, (noise.shape[0], 8*self.ngf, 4, 4))
        cond = tf.concat((noise, c), axis=1)
        for GBlock in self.GBlocks:
            out = GBlock(out, cond)
        out = self.to_rgb(out)
        return out

class NetD(Model):
    def __init__(self, ndf, imsize=128, ch_size=3):
        super(NetD, self).__init__()
        self.conv_img = layers.Conv2D(ndf, 3, 1, 'same')
        self.DBlocks = []
        in_out_pairs = get_D_in_out_chs(ndf, imsize)
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.DBlocks.append(D_Block(in_ch, out_ch))

    def call(self, x):
        out = self.conv_img(x)
        for DBlock in self.DBlocks:
            out = DBlock(out)
        return out

class NetC(Model):
    def __init__(self, ndf, cond_dim=256):
        super(NetC, self).__init__()
        self.cond_dim = cond_dim
        self.joint_conv = tf.keras.Sequential([
            layers.Conv2D(ndf*2, 3, 1, 'same', use_bias=False),
            layers.LeakyReLU(0.2),
            layers.Conv2D(1, 4, 1, 'valid', use_bias=False)
        ])

    def call(self, out, y):
        y = tf.reshape(y, (-1, self.cond_dim, 1, 1))
        y = tf.tile(y, (1, 1, 4, 4))
        h_c_code = tf.concat((out, y), axis=1)
        out = self.joint_conv(h_c_code)
        return out

class G_Block(layers.Layer):
    def __init__(self, cond_dim, in_ch, out_ch, upsample):
        super(G_Block, self).__init__()
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch
        self.c1 = layers.Conv2D(out_ch, 3, 1, 'same')
        self.c2 = layers.Conv2D(out_ch, 3, 1, 'same')
        self.fuse1 = DFBLK(cond_dim, in_ch)
        self.fuse2 = DFBLK(cond_dim, out_ch)
        if self.learnable_sc:
            self.c_sc = layers.Conv2D(out_ch, 1, 1, 'valid')

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, y):
        h = self.fuse1(h, y)
        h = self.c1(h)
        h = self.fuse2(h, y)
        h = self.c2(h)
        return h

    def call(self, x, y):
        if self.upsample:
            x = tf.image.resize(x, (x.shape[1]*2, x.shape[2]*2))
        return self.shortcut(x) + self.residual(x, y)

class D_Block(layers.Layer):
    def __init__(self, fin, fout, downsample=True):
        super(D_Block, self).__init__()
        self.downsample = downsample
        self.learned_shortcut = fin != fout
        self.conv_r = tf.keras.Sequential([
            layers.Conv2D(fout, 4, 2, 'same', use_bias=False),
            layers.LeakyReLU(0.2),
            layers.Conv2D(fout, 3, 1, 'same', use_bias=False),
            layers.LeakyReLU(0.2)
        ])
        self.conv_s = layers.Conv2D(fout, 1, 1, 'valid')
        self.gamma = tf.Variable(tf.zeros(1), trainable=True)

    def call(self, x):
        res = self.conv_r(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            x = tf.nn.avg_pool2d(x, 2, 2, 'VALID')
        return x + self.gamma * res

class DFBLK(layers.Layer):
    def __init__(self, cond_dim, in_ch):
        super(DFBLK, self).__init__()
        self.affine0 = Affine(cond_dim, in_ch)
        self.affine1 = Affine(cond_dim, in_ch)

    def call(self, x, y=None):
        h = self.affine0(x, y)
        h = layers.LeakyReLU(0.2)(h)
        h = self.affine1(h, y)
        h = layers.LeakyReLU(0.2)(h)
        return h

class Affine(layers.Layer):
    def __init__(self, cond_dim, num_features):
        super(Affine, self).__init__()
        self.fc_gamma = tf.keras.Sequential([
            layers.Dense(num_features),
            layers.ReLU(),
            layers.Dense(num_features)
        ])
        self.fc_beta = tf.keras.Sequential([
            layers.Dense(num_features),
            layers.ReLU(),
            layers.Dense(num_features)
        ])
        self._initialize()

    def _initialize(self):
        self.fc_gamma.layers[-1].kernel_initializer = Zeros()
        self.fc_gamma.layers[-1].bias_initializer = Ones()
        self.fc_beta.layers[-1].kernel_initializer = Zeros()
        self.fc_beta.layers[-1].bias_initializer = Zeros()

    def call(self, x, y=None):
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        if tf.rank(weight) == 1:
            weight = tf.expand_dims(weight, 0)
        if tf.rank(bias) == 1:
            bias = tf.expand_dims(bias, 0)

        size = tf.shape(x)
        weight = tf.broadcast_to(tf.expand_dims(tf.expand_dims(weight, -1), -1), size)
        bias = tf.broadcast_to(tf.expand_dims(tf.expand_dims(bias, -1), -1), size)
        return weight * x + bias

def get_G_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    in_out_pairs = list(zip(channel_nums[:-1], channel_nums[1:]))
    return in_out_pairs

def get_D_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    in_out_pairs = list(zip(channel_nums[:-1], channel_nums[1:]))
    return in_out_pairs