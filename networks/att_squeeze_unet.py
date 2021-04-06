import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, MaxPool2D, BatchNormalization, ReLU, LeakyReLU, UpSampling2D, Activation, ZeroPadding2D, Lambda, AveragePooling2D, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import Model, Sequential

  
class FireModule(tf.keras.Model):
    def __init__(self, fire_id, squeeze, expand):
        super(FireModule, self).__init__(name='')

        self.fire = Sequential()
        self.fire.add(Conv2D(squeeze, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal'))
        self.fire.add(BatchNormalization(axis=-1))
        self.left = Conv2D(expand, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal')
        self.right = Conv2D(expand, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')

    def call(self, x):
        x = self.fire(x)
        left = self.left(x)
        right = self.right(x)
        x = tf.concat([left, right], axis=-1)
        return x
    
class AttFireModule(tf.keras.Model):
    def __init__(self, filters, squeeze):
        super(AttFireModule, self).__init__(name='')

        self.fire = Sequential()
        self.fire.add(Conv2D(squeeze, (1, 1), strides=(1, 1), use_bias=False, data_format='channels_last', kernel_initializer='he_normal'))
        self.fire.add(BatchNormalization(axis=-1))
        self.left = Conv2D(filters, (1, 1), strides=(1, 1), use_bias=False, data_format='channels_last', kernel_initializer='he_normal')
        self.right = Conv2D(filters, (1, 1), strides=(1, 1), use_bias=False, data_format='channels_last', kernel_initializer='he_normal')

    def call(self, x):
        x = self.fire(x)
        left = self.left(x)
        right = self.right(x)
        x = tf.concat([left, right], axis=-1)
        return x
    
class AttentionBlock(Model):
    def __init__(self, filters):
        super(AttentionBlock, self).__init__(name='')
      
        self.w_g = Sequential()
        self.w_g.add(Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), use_bias=False, data_format='channels_last', kernel_initializer='he_normal'))
        self.w_g.add(BatchNormalization())

        self.w_x = Sequential()
        self.w_x.add(Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), use_bias=False, data_format='channels_last', kernel_initializer='he_normal'))
        self.w_x.add(BatchNormalization())

        self.psi = Sequential()
        self.psi.add(Conv2D(1, kernel_size=(1, 1), strides=(1, 1), use_bias=False, data_format='channels_last', kernel_initializer='he_normal'))
        self.psi.add(BatchNormalization())
        self.psi.add(Activation("sigmoid"))

        self.relu = ReLU()

    def call(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        out = x*psi
        return out

   
class UpsamplingBlock(tf.keras.Model):
    def __init__(self, filters, fire_id, squeeze, expand, strides, deconv_ksize, att_filters):
        super(UpsamplingBlock, self).__init__(name='')
        self.upconv = Conv2DTranspose(filters, deconv_ksize, strides=strides, padding='same', kernel_initializer='he_normal')
        self.fire = FireModule(fire_id, squeeze, expand)
        self.attention = AttentionBlock(att_filters)

    def call(self, x, g):
        d = self.upconv(x)
        x = self.attention(d, g)
        d = tf.concat([x, d], axis=-1)
        x = self.fire(d)    
        return x

class AttSqueezeUNet(Model):
    def __init__(self, n_classes=2, dropout=False): #filters equals to the number of classes
        super(AttSqueezeUNet, self).__init__(name='AttSqueezeUNet')
        self.__dropout = dropout
        channel_axis = -1
        self.conv_1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer='he_normal')
        self.max_pooling_1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')
        
        self.fire_1 = FireModule(2, 16, 64)
        self.fire_2 = FireModule(3, 16, 64)
        self.max_pooling_2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")
        
        self.fire_3 = FireModule(3, 32, 128)
        self.fire_4 = FireModule(4, 32, 128)
        self.max_pooling_3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")
        
        self.fire_5 = FireModule(5, 48, 192)
        self.fire_6 = FireModule(6, 48, 192)
        self.fire_7 = FireModule(7, 64, 256)
        self.fire_8 = FireModule(8, 64, 256)
        
        self.upsampling_1 = UpsamplingBlock(filters=192, fire_id=9, squeeze=48, expand=192, strides=(1, 1), deconv_ksize=3, att_filters=96)
        self.upsampling_2 = UpsamplingBlock(filters=128, fire_id=10, squeeze=32, expand=128, strides=(1, 1), deconv_ksize=3, att_filters=64)
        self.upsampling_3 = UpsamplingBlock(filters=64, fire_id=11, squeeze=16, expand=64, strides=(2, 2), deconv_ksize=3, att_filters=16)
        self.upsampling_4 = UpsamplingBlock(filters=32, fire_id=12, squeeze=16, expand=32, strides=(2, 2), deconv_ksize=3, att_filters=4)
        self.upsampling_5 = UpSampling2D(size=(2, 2))
        
        self.conv_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')
        self.upsampling_6 = UpSampling2D(size=(2, 2))
        self.conv_3 = Conv2D(n_classes, (1, 1), activation='softmax' if n_classes > 1 else 'sigmoid')
        
    def call(self, x):
        
        x0 = self.conv_1(x)
        x1 = self.max_pooling_1(x0)
        
        x2 = self.fire_1(x1)
        x2 = self.fire_2(x2)
        x2 = self.max_pooling_2(x2)
        
        x3 = self.fire_3(x2)
        x3 = self.fire_4(x3)
        x3 = self.max_pooling_3(x3)
        
        x4 = self.fire_5(x3)
        x4 = self.fire_6(x4)
        
        x5 = self.fire_7(x4)
        x5 = self.fire_8(x5)
        
        if self.__dropout:
            x5 = Dropout(0.2)(x5)
            
        d5 = self.upsampling_1(x5, x4)
        d4 = self.upsampling_2(d5, x3)
        d3 = self.upsampling_3(d4, x2)
        d2 = self.upsampling_4(d3, x1)
        d1 = self.upsampling_5(d2)
        
        d0 = tf.concat([d1, x0], axis=-1)
        d0 = self.conv_2(d0)
        d0 = self.upsampling_6(d0)
        d = self.conv_3(d0)
        
        return d
