import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, BatchNormalization, ReLU, LeakyReLU, UpSampling2D, Activation, ZeroPadding2D, Lambda, AveragePooling2D, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import Model, Sequential

class MaxPoolingWithArgmax2D(Model):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same"):
        super(MaxPoolingWithArgmax2D, self).__init__(name='')
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        ksize = [1, pool_size[0], pool_size[1], 1]
        padding = padding.upper()
        strides = [1, strides[0], strides[1], 1]
        output, argmax = tf.nn.max_pool_with_argmax(
            inputs, ksize=ksize, strides=strides, padding=padding
        )
        
        argmax = tf.cast(argmax, dtype=tf.float32)
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx] if dim is not None else None
            for idx, dim in enumerate(input_shape)
        ]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Model):
    def __init__(self, size=(2, 2)):
        super(MaxUnpooling2D, self).__init__(name='')
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        mask = tf.cast(mask, "int32")
        input_shape = updates.shape
        #  calculation new shape
        if output_shape is None:
            output_shape = (
                input_shape[0],
                input_shape[1] * self.size[0],
                input_shape[2] * self.size[1],
                input_shape[3],
            )
        self.output_shape1 = output_shape

        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(mask, dtype="int32")
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = tf.reshape(
            tf.range(output_shape[0], dtype="int32"), shape=batch_shape
        )
        b = one_like_mask * batch_range
        y = mask // (self.output_shape1[2] * self.output_shape1[3])
        x = (mask // self.output_shape1[3]) % self.output_shape1[2]
        feature_range = tf.range(self.output_shape1[3], dtype="int32")
        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, self.output_shape1)
        return ret

class ConvBlock(Model):
    def __init__(self, filters, iterations=2, kernel_size=(3, 3)):
        super(ConvBlock, self).__init__(name='')
        self.conv = Sequential()
        for _ in range(iterations):
            self.conv.add(Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding='same', data_format='channels_last', use_bias=True, kernel_initializer='he_normal'))
            self.conv.add(BatchNormalization())
            self.conv.add(LeakyReLU())

    def call(self, x):
        x = self.conv(x)
        return x


class Segnet(Model):
    def __init__(self, size=(512, 512), n_classes=2): #filters equals to the number of classes
        super(Segnet, self).__init__(name='Segnet')

        self.max_pooling_1 = MaxPoolingWithArgmax2D()
        self.max_pooling_2 = MaxPoolingWithArgmax2D()
        self.max_pooling_3 = MaxPoolingWithArgmax2D()
        self.max_pooling_4 = MaxPoolingWithArgmax2D()
        self.max_pooling_5 = MaxPoolingWithArgmax2D()

        
        self.conv_1 = ConvBlock(64, 2)
        self.conv_2 = ConvBlock(128, 2)
        self.conv_3 = ConvBlock(256, 3)
        self.conv_4 = ConvBlock(512, 3)
        self.conv_5 = ConvBlock(512, 3)


        self.upsampling_5 = MaxUnpooling2D()
        self.upsampling_conv_5 = ConvBlock(512, 3)

        self.upsampling_4 = MaxUnpooling2D()
        self.upsampling_conv_4_1 = ConvBlock(512, 2)
        self.upsampling_conv_4_2 = ConvBlock(256, 1)

        self.upsampling_3 = MaxUnpooling2D()
        self.upsampling_conv_3_1 = ConvBlock(256, 2)
        self.upsampling_conv_3_2 = ConvBlock(128, 1)

        self.upsampling_2 = MaxUnpooling2D()
        self.upsampling_conv_2_1 = ConvBlock(128, 1)
        self.upsampling_conv_2_2 = ConvBlock(64, 1)

        self.upsampling_1 = MaxUnpooling2D()
        self.upsampling_conv_1 = ConvBlock(64, 1)

        self.conv_1x1 = Conv2D(n_classes, kernel_size=1, strides=1, kernel_initializer='he_normal', data_format='channels_last', activation='softmax' if n_classes > 1 else 'sigmoid')
        
    
    def call(self, x):
        
        #encoding

        x1 = self.conv_1(x)
        x1, mask1 = self.max_pooling_1(x1)
        
        x2 = self.conv_2(x1)
        x2, mask2 = self.max_pooling_2(x2)

        x3 = self.conv_3(x2)
        x3, mask3 = self.max_pooling_3(x3)

        x4 = self.conv_4(x3)
        x4, mask4 = self.max_pooling_4(x4)

        x5 = self.conv_5(x4)
        x5, mask5 = self.max_pooling_5(x5)


        #decoding
        
        d5 = self.upsampling_5([x5, mask5])
        d5 = self.upsampling_conv_5(d5)
        
        d4 = self.upsampling_4([d5, mask4])
        d4 = self.upsampling_conv_4_1(d4)
        d4 = self.upsampling_conv_4_2(d4)

        d3 = self.upsampling_3([d4, mask3])
        d3 = self.upsampling_conv_3_1(d3)
        d3 = self.upsampling_conv_3_2(d3)

        d2 = self.upsampling_2([d3, mask2])
        d2 = self.upsampling_conv_2_1(d2)
        d2 = self.upsampling_conv_2_2(d2)
        

        d1 = self.upsampling_1([d2, mask1])
        d1 = self.upsampling_conv_1(d1)

        d0 = self.conv_1x1(d1)
        
        return d0
