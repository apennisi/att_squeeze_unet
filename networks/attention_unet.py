import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, MaxPool2D, BatchNormalization, ReLU, LeakyReLU, UpSampling2D, Activation, ZeroPadding2D, Lambda, AveragePooling2D, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import Model, Sequential

class ConvBlock(Model):
    def __init__(self, filters):
        super(ConvBlock, self).__init__(name='')
        self.conv = Sequential()
        self.conv.add(Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last', use_bias=True, kernel_initializer='he_normal'))
        self.conv.add(BatchNormalization())
        self.conv.add(ReLU())
        self.conv.add(Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last', use_bias=True, kernel_initializer='he_normal'))
        self.conv.add(BatchNormalization())
        self.conv.add(ReLU())        

    def call(self, x):
        x = self.conv(x)
        return x


class UpsamplingBlock(tf.keras.Model):
    def __init__(self, filters, size=(2,2)):
        super(UpsamplingBlock, self).__init__(name='')
        self.upconv = Sequential()
        self.upconv.add(Conv2DTranspose(filters, (1, 1), strides=(2, 2), data_format='channels_last', kernel_initializer='he_normal'))
        self.upconv.add(Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False, data_format='channels_last', kernel_initializer='he_normal'))
        self.upconv.add(BatchNormalization())
        self.upconv.add(ReLU())

    def call(self, x):
        x = self.upconv(x)
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

class AttentionUNet(Model):
    def __init__(self, n_classes=2, size=(512, 512)): #filters equals to the number of classes
        super(AttentionUNet, self).__init__(name='')

        self.max_pooling_1 = MaxPool2D(pool_size=2, strides=2)
        self.max_pooling_2 = MaxPool2D(pool_size=2, strides=2)
        self.max_pooling_3 = MaxPool2D(pool_size=2, strides=2)
        self.max_pooling_4 = MaxPool2D(pool_size=2, strides=2)

        self.conv_1 = ConvBlock(64) #256
        self.conv_2 = ConvBlock(128) #128
        self.conv_3 = ConvBlock(256) #64
        self.conv_4 = ConvBlock(512) #32
        self.conv_5 = ConvBlock(1024) #16
        
        self.upsampling_4 = UpsamplingBlock(512)
        self.attention_4 = AttentionBlock(256)
        self.upsampling_conv_4 = ConvBlock(512)

        self.upsampling_3 = UpsamplingBlock(256)
        self.attention_3 = AttentionBlock(128)
        self.upsampling_conv_3 = ConvBlock(256)

        self.upsampling_2 = UpsamplingBlock(128)
        self.attention_2 = AttentionBlock(64)
        self.upsampling_conv_2 = ConvBlock(128)

        self.upsampling_1 = UpsamplingBlock(64)
        self.attention_1 = AttentionBlock(32)
        self.upsampling_conv_1 = ConvBlock(64)

        self.conv_1x1 = Conv2D(n_classes, kernel_size=1, strides=1, kernel_initializer='he_normal', data_format='channels_last', activation='softmax' if n_classes > 1 else 'sigmoid')
        
        
    def call(self, x):

        #encoding
        x1 = self.conv_1(x)
        
        x2 = self.max_pooling_1(x1)
        x2 = self.conv_2(x2)

        x3 = self.max_pooling_2(x2)
        x3 = self.conv_3(x3)

        x4 = self.max_pooling_3(x3)
        x4 = self.conv_4(x4)

        x5 = self.max_pooling_4(x4)
        x5 = self.conv_5(x5)

        #decoding
        
        d5 = self.upsampling_4(x5)
        x4 = self.attention_4(d5, x4)
        d5 = tf.concat([x4, d5], axis=-1)
        d5 = self.upsampling_conv_4(d5)

        d4 = self.upsampling_3(d5)
        x3 = self.attention_3(d4, x3)
        d4 = tf.concat([x3, d4], axis=-1)
        d4 = self.upsampling_conv_3(d4)

        d3 = self.upsampling_2(d4)
        x2 = self.attention_2(d3, x2)
        d3 = tf.concat([x2, d3], axis=-1)
        d3 = self.upsampling_conv_2(d3)

        d2 = self.upsampling_1(d3)
        x1 = self.attention_1(d2, x1)
        d2 = tf.concat([x1, d2], axis=-1)
        d2 = self.upsampling_conv_1(d2)

        d1 = self.conv_1x1(d2)

        return d1
