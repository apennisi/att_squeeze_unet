import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

def jaccard_coef(y_true, y_pred, smooth=1., activation = False):

    y_true = K.permute_dimensions(y_true, (3,1,2,0))
    y_pred = K.permute_dimensions(y_pred, (3,1,2,0))
    
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    prediction = y_pred_f

    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true_f * prediction), axis=-1)
    sum_ = tf.keras.backend.sum(tf.keras.backend.abs(y_true_f) + tf.keras.backend.abs(prediction), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    result = tf.cond(jac < 0.65, lambda: 0., lambda: jac)
    return result


def tversky(y_true, y_pred, smooth=1, alpha=0.7):    
    y_true_pos = tf.keras.backend.flatten(y_true)
    y_pred_pos = tf.keras.backend.flatten(y_pred)
    product = tf.math.multiply(y_true_pos, y_pred_pos)
    true_pos = tf.keras.backend.sum(product)
    false_neg = tf.keras.backend.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.keras.backend.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg +
                                  (1 - alpha) * false_pos + smooth)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return tf.pow((1 - tv), gamma)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) +
                                           smooth)