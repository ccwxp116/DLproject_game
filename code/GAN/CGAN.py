import keras
from keras import layers
import numpy as np
import tensorflow as tf


class ConditionalGAN(keras.Model):
    def __init__(self, num_classes, img_shape=(64,64,3), latent_dim=200, **kwargs):
        super().__init__(**kwargs)
        self.dis_model = ConditionalDiscriminator()
        self.gen_model = ConditionalGenerator(latent_dim, num_classes, out_channel=img_shape[-1])
        self.latent_dim = latent_dim
        self.z_sampler = tf.random.normal
        
        self.num_classes = num_classes
        self.img_shape = img_shape


    def compile(self, optimizers, losses, accuracies, **kwargs):
        super().compile(
            loss = losses.values(),
            optimizer = optimizers.values(),
            metrics = accuracies.values(),
            **kwargs
        )
        self.loss_funcs = losses
        self.optimizers = optimizers
        self.acc_funcs = accuracies

    def fit(self, *args, d_steps=1, g_steps=1, **kwargs):
        self.d_steps = d_steps
        self.g_steps = g_steps
        super().fit(*args, **kwargs)

    def sample_z(self, n):
        '''generates an z based on the z sampler'''
        return self.z_sampler([n, self.latent_dim])

    def discriminate(self, imgs, class_labels, **kwargs):
        '''predict whether input input is a real entry from the true dataset'''
        class_labels_image = class_labels[:, None, None, :]
        class_labels_image = tf.repeat(class_labels_image, repeats=[self.img_shape[0] * self.img_shape[1]])
        class_labels_image = tf.reshape(class_labels_image, (-1, self.img_shape[0], self.img_shape[1], self.num_classes))
        
        return self.dis_model(tf.concat([imgs, class_labels_image], -1), **kwargs)

    def generate(self, n, class_label, **kwargs):
        '''generates an output based on a specific z realization'''
        z = self.sample_z(n)
        class_label = keras.utils.to_categorical(class_label, num_classes=self.num_classes)
        class_label = tf.expand_dims(class_label, 0)
        class_label = tf.repeat(class_label, repeats=[n], axis=0)
        z_with_labels = tf.concat([z, class_label], -1)
        
        return self.gen_model(z_with_labels, **kwargs)
        

    def test_step(self, data):
        real_imgs, class_labels = data
        batch_size = tf.shape(real_imgs)[0]

        z = self.sample_z(batch_size)
        fake_imgs = self.generate(z, class_labels)
        real_preds = self.dis_model(real_imgs, class_labels) # should be 1
        fake_preds = self.dis_model(fake_imgs, class_labels) # if generator is good, should be 1. Otherwise 0

        metrics = {}
        metrics['d_loss'] = self.loss_funcs['loss_func'](real_preds, tf.ones_like(real_preds))
        metrics['g_loss'] = self.loss_funcs['loss_func'](fake_preds, tf.ones_like(fake_preds))
        metrics['d_acc_real'] = self.acc_funcs['d_acc_real'](real_preds, tf.ones_like(real_preds))
        metrics['d_acc_fake'] = self.acc_funcs['d_acc_fake'](fake_preds, tf.zeros_like(fake_preds))
        metrics['g_acc'] = self.acc_funcs['g_acc'](fake_preds, tf.ones_like(fake_preds))
        return metrics


    def train_step(self, data):
        real_imgs, class_labels = data
        batch_size = tf.shape(real_imgs)[0]

        z = self.sample_z(batch_size)

        class_labels_image = class_labels[:, None, None, :]
        class_labels_image = tf.repeat(class_labels_image, repeats=self.img_shape[0], axis=1)
        class_labels_image = tf.repeat(class_labels_image, repeats=self.img_shape[1], axis=2)
        class_labels_image = tf.reshape(class_labels_image, (-1, self.img_shape[0], self.img_shape[1], self.num_classes))
        class_labels_image = tf.cast(class_labels_image, tf.float32)

        # train discriminator with real image
        d_loss_func = self.loss_funcs['loss_func']
        d_opt = self.optimizers['d_opt']

        z_with_labels = tf.concat([z, class_labels], -1)
        fake_imgs = self.gen_model(z_with_labels, training=True)
        fake_imgs_with_labels = tf.concat([fake_imgs, class_labels_image], -1)
        real_imgs_with_labels = tf.concat([real_imgs, class_labels_image], -1)
        real_fake_labels = tf.concat([tf.random.uniform((batch_size, 1),0,0.3), tf.random.uniform((batch_size, 1), 0.7, 1.2, dtype=tf.float32)], 0) # soft labels

        # 10% of the time, flip the labels (noisy labels)
        if np.random.rand() < 0.1:
            real_fake_labels = tf.reverse(real_fake_labels, [0])

        for _ in range(self.d_steps):
            with tf.GradientTape() as tape:
                fake_preds = self.dis_model(fake_imgs_with_labels, training=True)
                real_preds = self.dis_model(real_imgs_with_labels, training=True)
                d_loss = d_loss_func(real_fake_labels, tf.concat([fake_preds, real_preds], 0))

            grads = tape.gradient(d_loss, self.dis_model.trainable_weights)
            d_opt.apply_gradients(zip(grads, self.dis_model.trainable_weights))


        # train generator
        g_loss_fn = self.loss_funcs['loss_func']
        g_opt = self.optimizers['g_opt']
        
        for _ in range(self.g_steps):
            z = self.sample_z(batch_size)
            z_with_labels = tf.concat([z, class_labels], -1)
            g_labels = tf.ones((batch_size, 1))
            
            with tf.GradientTape() as tape:
                fake_imgs = self.gen_model(z_with_labels, training=True)
                fake_preds = self.dis_model(tf.concat([fake_imgs, class_labels_image], -1), training=False)
                g_loss = g_loss_fn(g_labels, fake_preds)
    
            grads = tape.gradient(g_loss, self.gen_model.trainable_weights)
            g_opt.apply_gradients(zip(grads, self.gen_model.trainable_weights))


        ##Compute final states for metric computation
        fake_imgs = self.gen_model(tf.concat([z, class_labels], -1), training=False)
        fake_preds = self.dis_model(tf.concat([fake_imgs, class_labels_image], -1), training=False)
        real_preds = self.dis_model(tf.concat([real_imgs, class_labels_image], -1), training=False)

        metrics = dict()
        metrics['d_loss'] = d_loss
        metrics['g_loss'] = g_loss
        metrics['d_acc_real'] = self.acc_funcs['d_acc_real'](real_preds, tf.ones_like(real_preds))
        metrics['d_acc_fake'] = self.acc_funcs['d_acc_fake'](fake_preds, tf.zeros_like(fake_preds))
        metrics['g_acc'] = self.acc_funcs['g_acc'](fake_preds, tf.ones_like(fake_preds))
        return metrics


class ConditionalGenerator(keras.Model):
    def __init__(self, latent_dim, num_classes, out_channel=3, dropout_rate=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.dense = layers.Dense(8 * 8 * (latent_dim + num_classes)) 
        self.reshape = layers.Reshape((8, 8, latent_dim + num_classes))  
        self.bn0 = layers.BatchNormalization()
        self.dropout0 = layers.Dropout(dropout_rate)
        self.prelu0 = layers.PReLU()
        
        self.deconv1 = layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding="same")
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.prelu1 = layers.PReLU()
        
        self.deconv2 = layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding="same")
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout_rate)
        self.prelu2 = layers.PReLU()
        
        self.deconv3 = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")
        self.bn3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(dropout_rate)
        self.prelu3 = layers.PReLU()
        
        self.deconv4 = layers.Conv2DTranspose(out_channel, (4, 4), strides=(1, 1), padding='same', activation='tanh')

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        x = self.bn0(x)
        x = self.reshape(x)
        x = self.dropout0(x, training=training)
        #x = tf.nn.leaky_relu(x, 0.2)
        x = self.prelu0(x)
        
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.dropout1(x, training=training)
        #x = tf.nn.leaky_relu(x, 0.2)
        x = self.prelu1(x)
        
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.dropout2(x, training=training)
        #x = tf.nn.leaky_relu(x, 0.2)
        x = self.prelu2(x)
        
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.dropout3(x, training=training)
        #x = tf.nn.leaky_relu(x, 0.2)
        x = self.prelu3(x)
        
        x = self.deconv4(x)
        return x



class ConditionalDiscriminator(keras.Model):
    def __init__(self):
        super().__init__()
        
        self.conv1 = layers.Conv2D(64, (3, 3), strides=(2, 2), padding="valid")
        self.dropout1 = layers.Dropout(0.5)
        self.prelu1 = layers.PReLU()
        
        self.conv2 = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="valid")
        self.dropout2 = layers.Dropout(0.5)
        self.prelu2 = layers.PReLU()
        
        self.conv3 = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")
        self.dropout3 = layers.Dropout(0.5)
        self.prelu3 = layers.PReLU()
        
        self.flatten = layers.Flatten()
        
        self.dense1 = layers.Dense(1000)
        self.prelu4 = layers.PReLU()
        self.dense2 = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False): 
        """inputs should be images combined with class labels, shape: (N, 64, 64, 1 + num_classes)"""
        x = self.conv1(inputs)
        x = self.dropout1(x, training=training)
        #x = tf.nn.leaky_relu(x)
        x = self.prelu1(x)
        
        x = self.conv2(x)
        x = self.dropout2(x, training=training)
        #x = tf.nn.leaky_relu(x)
        x = self.prelu2(x)
        
        
        x = self.conv3(x)
        x = self.dropout3(x, training=training)
        #x = tf.nn.leaky_relu(x)
        x = self.prelu3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        #x = tf.nn.leaky_relu(x)
        x = self.prelu4(x)
        
        x = self.dense2(x)
        return x