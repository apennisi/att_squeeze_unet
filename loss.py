import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

def jaccard_loss(y_true, y_pred, smooth=1., activation = False):
    return 1 - jaccard_coef(y_true, y_pred, smooth)

def jaccard_coef(y_true, y_pred, smooth=1., activation = False):

    y_true = K.permute_dimensions(y_true, (3,1,2,0))
    y_pred = K.permute_dimensions(y_pred, (3,1,2,0))
    
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    prediction = y_pred_f
    # prediction = tf.keras.layers.Activation('sigmoid')(prediction)
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true_f * prediction), axis=-1)
    sum_ = tf.keras.backend.sum(tf.keras.backend.abs(y_true_f) + tf.keras.backend.abs(prediction), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    result = tf.cond(jac < 0.65, lambda: 0., lambda: jac)
    return result

# def jaccard_coef(y_true, y_pred, smooth=1., activation = False):
#     d1, _, _, _ = tf.unstack(y_pred, axis=0)   
#     y_true_f = tf.keras.backend.flatten(y_true)
#     y_pred_f = tf.keras.backend.flatten(d1)
#     prediction = y_pred_f
#     # prediction = tf.keras.layers.Activation('sigmoid')(prediction)
#     intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true_f * prediction), axis=-1)
#     sum_ = tf.keras.backend.sum(tf.keras.backend.abs(y_true_f) + tf.keras.backend.abs(prediction), axis=-1)
#     jac = (intersection + smooth) / (sum_ - intersection + smooth)
#     result = tf.cond(jac < 0.65, lambda: 0., lambda: jac)
#     return result


def dice_coef(y_true, y_pred, smooth=1):
    # prediction = y_pred
    # prediction = tf.keras.layers.Activation('sigmoid')(prediction)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) +
                                           smooth)
    
def dice_coef_test(y_true, y_pred, smooth=1):
    d1, _, _, _, _ = tf.unstack(y_pred, axis=0) 
    # prediction = y_pred
    # prediction = tf.keras.layers.Activation('sigmoid')(prediction)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(d1)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) +
                                           smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def tversky(y_true, y_pred, smooth=1, alpha=0.7):    
    y_true_pos = tf.keras.backend.flatten(y_true)
    y_pred_pos = tf.keras.backend.flatten(y_pred)
    product = tf.math.multiply(y_true_pos, y_pred_pos)
    true_pos = tf.keras.backend.sum(product)
    false_neg = tf.keras.backend.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.keras.backend.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg +
                                  (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return tf.pow((1 - tv), gamma)



def class_tversky(y_true, y_pred):
    smooth = 1

    y_true = K.permute_dimensions(y_true, (3,1,2,0))
    y_pred = K.permute_dimensions(y_pred, (3,1,2,0))

    y_true_pos = K.batch_flatten(y_true)
    y_pred_pos = K.batch_flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos, 1)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
    false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def focal_tversky_loss_multi_class(y_true,y_pred):
    pt_1 = class_tversky(y_true, y_pred)
    gamma = 0.75
    return K.sum(K.pow((1-pt_1), gamma))




######################################################################
######################################################################

class TverskyError(tf.keras.losses.Loss):
    def __init__(self, size):
        super(TverskyError, self).__init__(name='')
        self.size = size
    
    def binary_image(self, result):
        result = result.reshape((self.size[0], self.size[1], 2)).argmax(axis=2)
        new_image = np.zeros((self.size[0], self.size[1]))
        for i in range(0, self.size[0]):
            for j in range(0, self.size[1]):
                if result[i, j] == 1:
                    new_image[i, j] = 1
                else:
                    pass
        return tf.convert_to_tensor(new_image, dtype=tf.float64)
    
    
    def tversky(self, y_true, y_pred, smooth=1, alpha=0.7):
        y_true_pos = tf.keras.backend.flatten(y_true)
        y_pred_pos = tf.keras.backend.flatten(y_pred)
        y_true_pos = tf.cast(y_true_pos, dtype=tf.float64)
        y_pred_pos = tf.cast(y_pred_pos, dtype=tf.float64)
        product = tf.math.multiply(y_true_pos, y_pred_pos)
        true_pos = tf.keras.backend.sum(product)
        false_neg = tf.keras.backend.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = tf.keras.backend.sum((1 - y_true_pos) * y_pred_pos)
        return (true_pos + smooth) / (true_pos + alpha * false_neg +
                                    (1 - alpha) * false_pos + smooth)
    
    def call(self, y_true, y_pred):
        d1, x1, x2, x3, x4 = tf.unstack(y_pred, axis=0) 
        gamma=0.75  
        tv_1 = self.tversky(y_true, x1)
        tv_2 = self.tversky(y_true, x2)
        tv_3 = self.tversky(y_true, x3)
        tv_4 = self.tversky(y_true, x4)
        
        tv_1 = tf.pow((1 - tv_1), gamma)
        tv_2 = tf.pow((1 - tv_2), gamma)
        tv_3 = tf.pow((1 - tv_3), gamma)
        tv_4 = tf.pow((1 - tv_4), gamma)
        
        loss = (tv_1 + tv_2 + tv_3 + tv_4) / 4
        
        return loss
    
class JaccardCoefMetric(tf.keras.metrics.Metric):
    
    def __init__(self, size):
        super(JaccardCoefMetric, self).__init__(name='')
        self.size = size
        self.jaccard = self.add_weight(name='jaccard', initializer='zeros')
        self.counter = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        d1, _, _, _, _ = tf.unstack(y_pred, axis=0)
        smooth=1.
        # d1 = tf.argmax(d1, axis=3)
        # y_true = tf.argmax(y_true, axis=3)
        y_true_f = tf.keras.backend.flatten(y_true)
        y_true_f = tf.cast(y_true_f, dtype=tf.float64)
        y_pred_f = tf.keras.backend.flatten(d1)
        prediction = tf.cast(y_pred_f, dtype=tf.float64)
        intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true_f * prediction), axis=-1)
        sum_ = tf.keras.backend.sum(tf.keras.backend.abs(y_true_f) + tf.keras.backend.abs(prediction), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        jac = tf.cond(jac < 0.65, lambda: 0., lambda: jac)
        self.counter += 1
        self.jaccard.assign_add(tf.cast(jac, dtype=tf.float32))

    def result(self):
        res = self.jaccard / self.counter
        self.counter = 0
        return res