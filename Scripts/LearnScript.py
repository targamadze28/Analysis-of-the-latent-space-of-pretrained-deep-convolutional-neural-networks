#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 15:01:20 2020

@author: vsevolod

"""

import numpy as np
import h5py as h5 

import os.path as fs

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, \
  SeparableConv2D, Dropout, concatenate , Conv2DTranspose, Layer
from keras.models import Model
from keras.datasets import mnist
from keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from keras.models import load_model


import keras
from keras.callbacks import TensorBoard
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split
#%%
def readHDF5file(PathToSave, SavedFileName, list_group_name):
  data = []
  ff = h5.File(fs.join(PathToSave, SavedFileName), 'r')
  for group in list_group_name:
    data.append(ff[group][...])
  ff.close()
  return data
#%%
"""
Image processing
"""

def normalization(images):
  max_value = np.max(images)
  images = images/max_value
  return images
  
def gauss_noise(images, mean, sigma):
  images += np.abs(np.random.uniform(mean, sigma, size = images.shape))
  return images
#%%
"""
Loss function
"""

def weighted_categorical_crossentropy(weights):
  """
  weighted_categorical_crossentropy
  weights --- веса к классам
  """
  weights = K.variable(weights)
  def loss(y_true, y_pred):
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss
  return loss

def categorical_focal_loss(alpha, gamma=2.):
  """
  categorical_focal_loss
  """
  alpha = np.array(alpha, dtype=np.float32)
  def categorical_focal_loss_fixed(y_true, y_pred):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * K.log(y_pred)
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
    return K.mean(K.sum(loss, axis=-1))
  return categorical_focal_loss_fixed

def dice_loss(y_true, y_pred):
  """
  dice_loss
  """
  smooth = 1.
  intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
  dice_coef = (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + \
                                      K.sum(K.square(y_pred),-1) + smooth)
  return 1 - dice_coef


def iou_loss_core(y_true, y_pred):
  """
  Intersection_over_union
  """
  smooth=1.
  intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
  union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
  iou = (intersection + smooth) / ( union + smooth)
  return iou

#%%
"""
Models
"""

def ezConvAutoEncoderForMnist():
  input_img = Input(shape=(64, 64, 3))  # adapt this if using `channels_first` image data format

  x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
  x = MaxPooling2D((2, 2))(x)
  x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2))(x)
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  encoded = MaxPooling2D((2, 2))(x)
  
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
  x = UpSampling2D((2, 2))(x)
  x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2, 2))(x)
  x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
  x = UpSampling2D((2, 2))(x)
  decoded = Conv2D(4, (3, 3), activation='softmax', padding='same')(x)
  autoencoder = Model(input_img, decoded)
  return autoencoder


def CustomConvAutoEncoderForMnist():
  input_img = Input(shape=(64, 64, 3))  # adapt this if using `channels_first` image data format

  x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
  x = MaxPooling2D((2, 2))(x)
  x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2))(x)
  x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  encoded = MaxPooling2D((2, 2))(x)
  
  x = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu')(encoded)
  x = Conv2DTranspose(16, (3, 3), strides=(2, 2), activation='relu')(x)
  x = Conv2DTranspose(8, (3, 3), strides=(2, 2), activation='relu')(x)
  x = Conv2D(8, (6, 6))(x)
  decoded = Conv2D(4, (3, 3), activation='softmax')(x)
  autoencoder = Model(input_img, decoded)
  return autoencoder


def Unet():
  input_img = Input(shape=(64, 64, 3)) 
  conv1 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_img)
  conv1 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  conv2 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
  conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
  conv4 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
  drop4 = Dropout(0.5)(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

  conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
  conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  drop5 = Dropout(0.5)(conv5)
  
  upp_tmp1 = UpSampling2D(size = (2,2))(drop5)
  up6 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upp_tmp1)
  merge6 = concatenate([drop4,up6], axis = 3)
  conv6 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  conv6 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
  
  upp_tmp2 = UpSampling2D(size = (2,2))(conv6)
  up7 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upp_tmp2)
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
  conv7 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

  upp_tmp3 = UpSampling2D(size = (2,2))(conv7)
  up8 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upp_tmp3)
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
  conv8 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
  
  upp_tmp4 = UpSampling2D(size = (2,2))(conv8)
  up9 = Conv2D(4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upp_tmp4)
  merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
  conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  conv9 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  conv10 = Conv2D(4, 1, activation = 'softmax')(conv9)
  autoencoder = Model(inputs=input_img, outputs=conv10)
  return autoencoder
  

def UnetСircumcised():
  input_img = Input(shape=(64, 64, 3)) 
  conv1 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_img)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
  drop4 = Dropout(0.5)(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

  conv5 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
  
  upp_tmp1 = UpSampling2D(size = (2,2))(conv5)
  up6 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upp_tmp1)
  merge6 = concatenate([drop4,up6], axis = 3)
  conv6 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  
  upp_tmp2 = UpSampling2D(size = (2,2))(conv6)
  up7 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upp_tmp2)
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)

  upp_tmp3 = UpSampling2D(size = (2,2))(conv7)
  up8 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upp_tmp3)
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
  
  upp_tmp4 = UpSampling2D(size = (2,2))(conv8)
  up9 = Conv2D(4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upp_tmp4)
  merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
  conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  conv10 = Conv2D(4, 1, activation = 'softmax')(conv9)
  autoencoder = Model(inputs=input_img, outputs=conv10)
  return autoencoder

def UnetWithSeparableConv():
  input_img = Input(shape=(64, 64, 3)) 
  conv1 = SeparableConv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_img)
  conv1 = SeparableConv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = SeparableConv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  conv2 = SeparableConv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = SeparableConv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
  conv3 = SeparableConv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = SeparableConv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
  conv4 = SeparableConv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
  drop4 = Dropout(0.5)(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
  
  conv5 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
  conv5 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  drop5 = Dropout(0.5)(conv5)
  
  upp_tmp1 = UpSampling2D(size = (2,2))(drop5)
  up6 = SeparableConv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upp_tmp1)
  merge6 = concatenate([drop4,up6], axis = 3)
  conv6 = SeparableConv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  conv6 = SeparableConv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
  
  upp_tmp2 = UpSampling2D(size = (2,2))(conv6)
  up7 = SeparableConv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upp_tmp2)
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = SeparableConv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
  conv7 = SeparableConv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
  
  upp_tmp3 = UpSampling2D(size = (2,2))(conv7)
  up8 = SeparableConv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upp_tmp3)
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = SeparableConv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
  conv8 = SeparableConv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
  
  upp_tmp4 = UpSampling2D(size = (2,2))(conv8)
  up9 = SeparableConv2D(4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upp_tmp4)
  merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = SeparableConv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
  conv9 = SeparableConv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  conv9 = SeparableConv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  conv10 = SeparableConv2D(4, 1, activation = 'softmax')(conv9)
  autoencoder = Model(inputs=input_img, outputs=conv10)
  return autoencoder

def UnetWithSeparableConvСircumcised():
  input_img = Input(shape=(64, 64, 3)) 
  conv1 = SeparableConv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_img)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  conv2 = SeparableConv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  conv3 = SeparableConv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  conv4 = SeparableConv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
  drop4 = Dropout(0.5)(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
  
  conv5 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
  drop5 = Dropout(0.5)(conv5)
  
  upp_tmp1 = UpSampling2D(size = (2,2))(drop5)
  up6 = SeparableConv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upp_tmp1)
  merge6 = concatenate([drop4,up6], axis = 3)
  conv6 = SeparableConv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  
  upp_tmp2 = UpSampling2D(size = (2,2))(conv6)
  up7 = SeparableConv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upp_tmp2)
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = SeparableConv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
  
  upp_tmp3 = UpSampling2D(size = (2,2))(conv7)
  up8 = SeparableConv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upp_tmp3)
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = SeparableConv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
  
  upp_tmp4 = UpSampling2D(size = (2,2))(conv8)
  up9 = SeparableConv2D(4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(upp_tmp4)
  merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = SeparableConv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
  conv9 = SeparableConv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  conv9 = SeparableConv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  conv10 = SeparableConv2D(4, 1, activation = 'softmax')(conv9)
  autoencoder = Model(inputs=input_img, outputs=conv10)
  return autoencoder


def AlexNet():
  input_img = Input(shape=(64, 64, 3)) 

  conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
  conv1 = Dropout(0.2)(conv1)
  conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
  pool1 = MaxPooling2D((2, 2))(conv1)
  
  conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = Dropout(0.2)(conv2)
  conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
  pool2 = MaxPooling2D((2, 2))(conv2)
  
  conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = Dropout(0.2)(conv3)
  conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)

  up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
  conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
  conv4 = Dropout(0.2)(conv4)
  conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv4)
  
  up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
  conv5 = Conv2D(16, (3, 3), activation='relu', padding='same')(up2)
  conv5 = Dropout(0.2)(conv5)
  conv5 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv5)
  
  out = Conv2D(4, (1, 1) , padding='same')(conv5)
  autoencoder = Model(inputs=input_img, outputs=out)
  return autoencoder

def AlexNetSeparable():
  input_img = Input(shape=(64, 64, 3)) 

  conv1 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(input_img)
  conv1 = Dropout(0.2)(conv1)
  conv1 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(conv1)
  pool1 = MaxPooling2D((2, 2))(conv1)
  
  conv2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(pool1)
  conv2 = Dropout(0.2)(conv2)
  conv2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(conv2)
  pool2 = MaxPooling2D((2, 2))(conv2)
  
  conv3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(pool2)
  conv3 = Dropout(0.2)(conv3)
  conv3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(conv3)

  up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
  conv4 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(up1)
  conv4 = Dropout(0.2)(conv4)
  conv4 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(conv4)
  
  up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
  conv5 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(up2)
  conv5 = Dropout(0.2)(conv5)
  conv5 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(conv5)
  
  out = SeparableConv2D(4, (1, 1) , padding='same')(conv5)
  autoencoder = Model(inputs=input_img, outputs=out)
  return autoencoder
#%%

PathToDataSet = '  '
NameDataSet = '  '
NameTestDataSet = '  '

PathToModel = ' '

fileData = readHDF5file(PathToDataSet, NameDataSet,\
                        ['image', 'mask'])

images = fileData[0].astype(np.float32)
mask = fileData[1].astype(np.float32)

fileData = readHDF5file(PathToDataSet, NameTestDataSet,\
                        ['image', 'mask'])

images_test = fileData[0].astype(np.float32)
mask_test = fileData[1].astype(np.float32)

images = gauss_noise(images, 0, 5)
images = normalization(images)

images_test = gauss_noise(images_test, 0, 5)
images_test = normalization(images_test)

weights_CE = np.ones((4,))
weights_CE[:3] = 4

NN_list = [ezConvAutoEncoderForMnist(),\
           DymasUnetСircumcised(),\
           DymasUnetWithSeparableConvСircumcised()]
 
loss_list = [weighted_categorical_crossentropy(weights_CE),\
             dice_loss]
    
name_loss_list = ['weighted_categorical_crossentropy',\
             'dice_loss']
  
name_NN_list = ['ezConvAutoEncoderForMnist', 'UnetСircumcised',\
             'UnetWithSeparableConvСircumcised']

NN_index = 2
for iter_NN in range(5):
  for iter_loss in range(len(loss_list)):
    
    print('NN = %s, loss = %s'%(name_NN_list[NN_index], name_loss_list[iter_loss]))
    
    autoencoder = NN_list[NN_index]
    autoencoder.compile(optimizer='Nadam', loss=loss_list[iter_loss])
    history = autoencoder.fit(images, mask, batch_size=4096, epochs=25, verbose=2, \
                          validation_data=(images_test, mask_test), shuffle=True)

    optim = keras.optimizers.SGD(lr=0.02, decay=1e-2/50)
    autoencoder.compile(optimizer=optim, loss=loss_list[iter_loss])
    history = autoencoder.fit(images, mask, batch_size=4096, epochs=50, verbose=2,\
                          validation_data=(images_test, mask_test), shuffle=True)
    autoencoder.save(fs.join(PathToModel, 'model_%s_%s_launch%d.hdf5'%(name_NN_list[NN_index], name_loss_list[iter_loss], iter_NN + 1)))

