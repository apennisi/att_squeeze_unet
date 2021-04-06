import argparse
from glob import glob

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam, SGD
 
from networks.attention_unet import AttentionUNet
from networks.segnet import Segnet
from networks.unet import UNet
from networks.squeeze_unet import SqueezeUNet
from networks.att_squeeze_unet import AttSqueezeUNet
from utils import *
from loss import *
import os


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

tf.config.run_functions_eagerly(True)
print(tf.executing_eagerly())
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
parser.add_argument('--network', dest='network', type=str, default="attention_segnet", help='Select network: attention_squeeze_unet, squeeze_unet, attention_unet, unet, segnet')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--aug_scale', dest='aug_scale', type=int, default=3, help='scale of data augmentation')
parser.add_argument("--resume", help="path to the model to resume")
parser.add_argument('--lr_decay', dest='lr_decay', default='time', help='time or exp')

parser.add_argument('--train_set', dest='train_set', help='training data path')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='./logs', help='tensorboard logs are saved here')
parser.add_argument('--eval_set', dest='eval_set', help='dataset for eval in training')

args = parser.parse_args()


def main():
    train_imgs = sorted(glob(args.train_set+"/*.jpg"))
    train_maps = sorted(glob(args.train_set+"/*.png"))

    train_steps_per_epoch = int(len(train_imgs) / args.batch_size)
    
    #validation set filepaths
    validation_imgs = sorted(glob(args.eval_set+"/*.jpg"))
    validation_maps = sorted(glob(args.eval_set+"/*.png"))

    val_steps_per_epoch = int(len(validation_imgs) / args.batch_size)
    size = (384, 512)
    train_gen = data_generator(
        train_imgs,
        train_maps,
        args.batch_size,
        args.aug_scale,
        size=size
    )

    val_gen = data_generator(
        validation_imgs,
        validation_maps,
        args.batch_size,
        args.aug_scale,
        validation=True,
        size=size
    )
    
    model = None
    if args.network == "attention_unet":
        model = AttentionUNet(size=size)
    elif args.network == "attention_squeeze_unet":
    	model = AttSqueezeUNet()
    elif args.network == "squeeze_unet":
        model = SqueezeUNet()
    elif args.network == "segnet":
        model = Segnet(size=size)
    elif args.network == "unet":
        model = UNet(size=size)
    else:
        raise ValueError("Network " + args.network + " unknown!")
    
    model.build(input_shape=(args.batch_size, size[1], size[0], 3))
    model.summary()
    model.compile(loss=focal_tversky_loss, optimizer=Adam(lr=args.lr), metrics=[jaccard_coef])


    if args.resume:
        print("Load Model: " + args.resume)
        model.load_weights(args.resume)

    
    initial_learning_rate = args.lr 
    
    lr_callback = None
    
    if args.lr_decay == 'exp':
        def lr_exp_decay(epoch, lr):
            import math
            k = 0.1
            return initial_learning_rate * math.exp(-k*epoch)
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=1)
    elif args.lr_decay == 'time':
        epochs = args.epoch
        decay = initial_learning_rate / epochs

        def lr_time_based_decay(epoch, lr):
            return lr * 1 / (1 + decay * epoch)
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_time_based_decay, verbose=1)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_jaccard_coef', patience=2, verbose=1)
    reduce_plateau = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1,
                      monitor='val_jaccard_coef',
                      patience=25,
                      min_lr=0.0000001,
                      verbose=1,
                      min_delta=0.0001,
                      mode='max')

    csvlogger = tf.keras.callbacks.CSVLogger("logs/training.log", separator=',', append=True)
    terminator = tf.keras.callbacks.TerminateOnNaN()
    
    

    filepath = args.ckpt_dir + "att_unet-{epoch:02d}-{val_jaccard_coef:.2f}.hdf5"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc',verbose=1, mode='max') 
    
    model.fit_generator(
        train_gen,
        steps_per_epoch=train_steps_per_epoch,
        epochs=args.epoch,
        validation_data=val_gen,
        validation_steps=val_steps_per_epoch,
        callbacks=[model_checkpoint, csvlogger, terminator, lr_callback, reduce_plateau]
    )


if __name__ == "__main__":
    main()
