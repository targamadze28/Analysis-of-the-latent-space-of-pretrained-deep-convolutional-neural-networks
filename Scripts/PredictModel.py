#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import h5py as h5 

import os.path as fs

import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, \
  SeparableConv2D, Dropout, concatenate , Conv2DTranspose, Layer
from keras import backend as K
from keras.models import Model, import load_model
from tensorflow.python.keras import Sequential

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
  weights to classes
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
  intersection over union loss
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
#%%

PathToDataSet = ''
NameDataSet = ''
NameTestDataSet = ''
PathToModel = ''

fileData = readHDF5file(PathToDataSet, NameTestDataSet,\
                        ['image', 'mask'])

images = fileData[0].astype(np.float32)
mask = fileData[1].astype(np.float32)

images = gauss_noise(images, 0, 5)
images = normalization(images)

weights_CE = np.ones((4,))
weights_CE[:3] = 4

weights_focal = np.ones((4,))*0.25
weights_focal[:3] = 0.75
  
NN_list = []
loss_list = []
metrics = []
  
name_loss_list = ['']
  
name_NN_list = ['']

precision = np.empty((len(NN_list), len(loss_list), 2), dtype = np.float32)
namemodel = np.empty((len(NN_list), len(loss_list)))
for iter_NN in range(len(NN_list)):
  for iter_loss in range(len(loss_list)):
    model = load_model(fs.join('',\
                               'model_%s_%s.hdf5'%(name_NN_list[iter_NN], name_loss_list[iter_loss])),\
                   custom_objects = {'loss':loss_list[iter_loss], 'weighted_categorical_crossentropy':weighted_categorical_crossentropy(weights_CE),\
                                     'dice_loss':dice_loss,\
                                       'categorical_focal_loss_fixed':categorical_focal_loss(weights_focal),\
                                         'iou_loss_core':iou_loss_core})
    result = model.evaluate(images, mask, batch_size=128)
    precision[iter_NN, iter_loss, 0] = 1 - result[-1]
    precision[iter_NN, iter_loss, 1] = 1 - result[-2]
    namemodel[iter_NN, iter_loss] = 'model_%s_%s.hdf5'%(name_NN_list[iter_NN], name_loss_list[iter_loss])

ff = h5.File(fs.join('', 'predict_result.hdf5'))
ff.create_dataset('precision', data = precision)
ff.create_dataset('namemodel', data = namemodel)
ff.close()

"""
Models Predictions
"""

namemodel = np.empty((len(NN_list), len(loss_list)), np.chararray)
for iter_NN in range(len(NN_list)):
  for iter_loss in range(len(loss_list)):
    namemodel[iter_NN, iter_loss] = 'model_%s_%s.hdf5'%(name_NN_list[iter_NN], name_loss_list[iter_loss])


for iter_NN in range(len(NN_list)):
  for iter_loss in range(len(loss_list)):
    print('name_model = ', namemodel[iter_NN, iter_loss], 'IoU = ', pred[iter_NN, iter_loss, 0],\
          'dice = ', pred[iter_NN, iter_loss, 1])
    print();print()


model = load_model(fs.join('',\
                           ''),\
                   custom_objects = {'loss':loss_list[-1], 'weighted_categorical_crossentropy':weighted_categorical_crossentropy(weights_CE),\
                                     'dice_loss':dice_loss,\
                                       'categorical_focal_loss_fixed':categorical_focal_loss(weights_focal),\
                                         'iou_loss_core':iou_loss_core})


pred = model.predict(images[:100])
  
ff = h5.File(fs.join(''), 'r')
pred = ff['precision'][...]
ff.close()

name_loss_list = ['']
name_NN_list = ['']

for i, nn_name in enumerate(name_NN_list):
  for j, loss_name in enumerate(name_loss_list):
    print(nn_name, loss_name, pred[i, j])
