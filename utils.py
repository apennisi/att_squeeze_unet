from __future__ import division
import numpy as np
from PIL import Image
import cv2 as cv
import tensorflow as tf

def load_images_RGB_float32(path, size = (512, 512), hr = False, hsv = False, gray = False, multi=False):
    im = cv.imread(path, cv.IMREAD_COLOR)
    if hsv:
        im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
    if hr:
        hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
        b, g, r = cv.split(im)
        h, s, v = cv.split(hsv)
        im = np.dstack((h, r))
    if gray:
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    
    if multi:
        hsvi = cv.cvtColor(im, cv.COLOR_BGR2HSV)
        lab = cv.cvtColor(im, cv.COLOR_BGR2Lab)
        l, _, _ = cv.split(lab)
        im = np.concatenate((im, hsvi, np.expand_dims(l, axis=3)), axis=-1)

    shape = im.shape
    im_size = (shape[1], shape[0])
    im_512 = cv.resize(im, (size[1], size[0]), interpolation=cv.INTER_AREA)
    im_512_np = None
    if gray:
        im_512_np = np.expand_dims(np.expand_dims(im_512, axis=3), axis=0).astype(np.float32) / 255.0
    else:
        im_512_np = np.expand_dims(im_512, axis=0).astype(np.float32) / 255.0
    im_np = np.expand_dims(np.array(im), axis=0).astype(np.float32) / 255.0
    return im_512_np, im_np, im_size


def category_label(labels, dims=(512, 512), n_labels=2):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            x[i, j, int(labels[i][j])] = 1
    return x

def test_generator(imglist, maplist, size=(512, 512), hr = False, hsv = False, gray = False, multi=False):
    assert len(imglist) == len(maplist)
    channels = None
    if hr:
        channels = 2
    elif gray:
        channels = 1
    elif multi:
        channels = 7
    else:
        channels = 3

    h_img, w_img = size
            
    for img, map in zip(imglist, maplist):
        img_array = np.zeros((1, h_img, w_img, channels), dtype='float32')
        map_array = np.zeros((1, h_img, w_img, 2), dtype='float32')
        im = cv.imread(img, cv.IMREAD_COLOR)
        mp = cv.imread(map, cv.IMREAD_GRAYSCALE)
        mp = np.where(mp == 255, 255, 0)
        shape = im.shape
        im_size = (shape[1], shape[0])
        if im_size != size: 
            im = cv.resize(im, (w_img, h_img), interpolation=cv.INTER_AREA)
            mp = cv.resize(mp.astype("uint8"), (w_img, h_img), interpolation=cv.INTER_AREA)

        if hr:
            hsv_im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
            b, g, r = cv.split(im)
            h, s, v = cv.split(hsv_im)
            im = np.dstack((h, r))
        
        if multi:
            hsvi = cv.cvtColor(im, cv.COLOR_BGR2HSV)
            lab = cv.cvtColor(im, cv.COLOR_BGR2Lab)
            l, _, _ = cv.split(lab)
            im = np.concatenate((im, hsvi, np.expand_dims(l, axis=3)), axis=-1)

        if hsv:
            im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
        
        if gray:
            im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        if gray:
            img_array[0] = np.expand_dims(np.expand_dims(im, axis=3), axis=0).astype(np.float32) / 255.0
        else:    
            img_array[0]= np.expand_dims(im, axis=0).astype(np.float32) / 255.0

        mp = mp.astype(np.float32) / 255.
        mp = category_label(mp, dims=size)
        map_array[0] = mp.astype(np.float32)
        yield img_array, map_array


def data_generator(imglist, maplist, batchsize, augment_scale, size = (512, 512), hr = False, hsv = False, gray = False, multi=False, validation = False):
    assert len(imglist) == len(maplist)
    while True:
        channels = None
        if hr:
            channels = 2
        elif gray:
            channels = 1
        elif multi:
            channels = 7
        else:
            channels = 3

        h_img, w_img = size

        img_array = np.zeros((batchsize, h_img, w_img, channels), dtype='float32')
        map_array = np.zeros((batchsize, h_img, w_img, 2), dtype='float32')
        
        ix = np.random.choice(np.arange(len(imglist)), batchsize)
        idx = 0
        for i in ix:
            im = cv.imread(imglist[i], cv.IMREAD_COLOR)
            mp = cv.imread(maplist[i], cv.IMREAD_GRAYSCALE)
            hsv_im = None
            
            shape = im.shape
            im_size = (shape[1], shape[0])

            if im_size != size: 
                im = cv.resize(im, (w_img, h_img), interpolation=cv.INTER_CUBIC)
                mp = cv.resize(mp, (w_img, h_img), interpolation=cv.INTER_CUBIC)
            
            if validation:
                if hr:
                    hsv_im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
                if hsv:
                    im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
                if gray:
                    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            else:
                #random data augmentation
                random_mode = np.random.randint(0, augment_scale + 1)
                im = data_augmentation(np.array(im), random_mode)
                mp = data_augmentation(np.array(mp), random_mode)
                
            if hr:
                hsv_im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
                
            if hsv:
                im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
            
            if gray:
                im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            
            if hr:
                b, g, r = cv.split(im)
                h, s, v = cv.split(hsv_im)
                im = np.dstack((h, r))

            if multi:
                hsvi = cv.cvtColor(im, cv.COLOR_BGR2HSV)
                lab = cv.cvtColor(im, cv.COLOR_BGR2Lab)
                l, _, _ = cv.split(lab)
                im = np.concatenate((im, hsvi, np.expand_dims(l, axis=-1)), axis=-1)
            
            if hr:
                img_array[idx]= np.expand_dims(im, axis=0).astype(np.float32) / 255.0
            elif gray:
                img_array[idx]= np.expand_dims(np.expand_dims(im, axis=3), axis=0).astype(np.float32) / 255.0
            else:
                img_array[idx]= np.expand_dims(im, axis=0).astype(np.float32) / 255.0
            
            mp = mp.astype(np.float32) / 255.
            mp = category_label(mp, dims=size)
            map_array[idx] = mp.astype(np.float32)
            
            idx = idx + 1

        yield img_array, map_array


#applies data augmentation according to the mode
def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.fliplr(image)
    elif mode == 3:
        img = np.flipud(image)
        img = np.fliplr(img)
        return img
    elif mode == 4:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 5:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 6:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 7:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 8:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)
