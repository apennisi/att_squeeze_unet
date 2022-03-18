import argparse
from networks.attention_unet import AttentionUNet
from networks.segnet import Segnet
from networks.unet import UNet
from networks.squeeze_unet import SqueezeUNet
from networks.att_squeeze_unet import AttSqueezeUNet
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Input
from utils import *
from loss import *
import random
import cv2
import cv2 as cv
import numpy as np

tf.config.run_functions_eagerly(True)
print(tf.executing_eagerly())
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)

def argparser():
    # command line argments
    parser = argparse.ArgumentParser(description="Attention Squeeze U-Net")
    parser.add_argument('--network', dest='network', type=str, default="attention_squeeze_unet", help='Select network: attention_squeeze_unet, squeeze_unet, attention_unet, unet, segnet')
    parser.add_argument("--test_dir", help="train test list path")
    parser.add_argument("--resume", help="path to the model to resume")
    parser.add_argument("--save_dir", help="output directory")
    args = parser.parse_args()

    return args

def color_image(result, size):
    result = result.reshape((size[0], size[1], 2)).argmax(axis=2)
    new_image = np.zeros((size[0], size[1]))
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if result[i, j] == 1:
                new_image[i, j] = 255
            else:
                pass
    return new_image

def main(args):

    from glob import glob
    list_images = sorted(glob(args.test_dir+"/*.jpg"))
    print("Number of images {n}".format(n=len(list_images)))
    assert len(list_images) != 0, "Error the testing image array is empty!"
    assert len(list_images) == len(list_images), "Error the testing image number differs from the number of masks"
        
    size = (384, 512)
    
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
    
    model.build(input_shape=(1, size[1], size[0], 3))    
    model.load_weights(args.resume)
    
    for image in list_images:
        im_512_np, im_np, im_size = load_images_RGB_float32(image, size=size)
        rgb = cv.imread(image, cv.IMREAD_COLOR)
        prediction = model.predict(im_512_np)
        pred_image = color_image(prediction, size=size)
        pred_image = cv.resize(pred_image, im_size, interpolation=cv.INTER_CUBIC)
        
        last_slash = image.rfind('/')
        name = image[last_slash+1:]
        name = name[:name.find('.')]
        print(image, name)        
        cv.imwrite(args.save_dir + "/" + name + ".png", pred_image) 
    

if __name__ == "__main__":
    args = argparser()
    main(args)
