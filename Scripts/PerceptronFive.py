#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:03:50 2020

@author: vsevolod
"""

import h5py as h5
import numpy as np

import matplotlib.pyplot as plt

import os.path as fs
import umap

import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score
from scipy.spatial import distance
from  sklearn.preprocessing import normalize

from sklearn.metrics import accuracy_score

import logging as logg

from sklearn.decomposition import PCA
from sklearn import preprocessing

#%%

def read_file(file_path, file_name, folder_name):
    with h5.File(fs.join(file_path, file_name), 'r') as f:
        data = f[folder_name][...]
    return data

def readHDF5file(PathToSave, SavedFileName, list_group_name):
  data = []
  ff = h5.File(fs.join(PathToSave, SavedFileName), 'r')
  for group in list_group_name:
    data.append(ff[group][...])
  ff.close()
  return data

def saveHDF5file(PathToSave, SavedFileName, list_group_name, data):
  num_group = len(list_group_name)
  num_data = len(data)
  if num_group != num_data:
   raise RuntimeError('Список имен групп и длина списка данных не соответствуют!')
  
  ff = h5.File(fs.join(PathToSave, SavedFileName), 'w')
  for i, group in enumerate(list_group_name):
    ff.create_dataset(group, data = data[i])
  ff.close()
  return None

def pca_result(activations, n_comp):
  embedding = PCA(n_components= n_comp).fit_transform(activations)
  return embedding

def umap_result(activations, n_comp):
    embedding = umap.UMAP(n_components=n_comp).fit_transform(activations)
    return embedding
  
#%%
   
def TrainPerceptron(latent_space):
  n = len(latent_space)
  shape_ls = latent_space.shape[1]
  print(shape_ls)
  labels = np.empty((n, 1), dtype = np.int32)
  labels[:15000], labels[15000:30000], labels[30000:] = 0, 1, 2
  #labels[:5000], labels[5000:10000], labels[10000:] = 0, 1, 2
  y_train = np_utils.to_categorical(labels)
  standardized_latent_space = preprocessing.scale(latent_space)
  
  model = Sequential()
  model.add(Dense(3, input_dim= shape_ls))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='Nadam')
  model.summary()
  model.fit(standardized_latent_space, y_train, epochs = 250, batch_size=128, validation_split=0.3, shuffle = True, verbose=2)
  optim = keras.optimizers.SGD(lr=0.02, decay=1e-2/300)
  model.compile(loss='categorical_crossentropy', optimizer=optim)
  model.fit(standardized_latent_space, y_train, epochs = 300, batch_size=128, validation_split=0.3, shuffle = True, verbose=2)


  predict = model.predict(standardized_latent_space, batch_size=4096)
  predict = np.heaviside(predict - 0.5, 1).astype(np.int32)
  score = f1_score(y_train, predict, average='micro')
  return score
  
#%%
RootPathLatentSpace = '/data/vsevolod/DmiDiplom/LatentSpace2T/preceptron'

logg.basicConfig(filename=fs.join(RootPathLatentSpace, "LatentSpaceLogger.log"), level=logg.INFO)
logg

umap_shape_mnist = [0, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
umap_shape_unet = [0, 512, 256, 128, 64, 32, 16, 8, 4, 2]

PathToDataSet = '/data/vsevolod/DmiDiplom/DataSets'
PathToModel = '/data/vsevolod/DmiDiplom/Models/gridNN'

NamesDataSet = ['OnlyColor.hdf5',\
                'OnlyH.hdf5',\
                'OnlyX.hdf5',\
                'Only.hdf5']
      
name_loss_list = ['weighted_categorical_crossentropy',\
             'dice_loss']
  
name_NN_list = ['ezConvAutoEncoderForMnist', 'DymasUnetСircumcised',\
             'DymasUnetWithSeparableConvСircumcised']

num_layers = [[6, 7], [7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12, 13]]

precision = np.ones((len(name_NN_list), len(name_loss_list), 5, len(NamesDataSet),\
                      7, 11, 2), dtype = np.float32)
precision = precision*(-1.)
print(precision)
for iter_NN in range(len(name_NN_list)):
  for iter_loss in range(len(name_loss_list)):
    for launch_num in range(5):        
      for data_iter, data in enumerate(NamesDataSet):
        number_layer = num_layers[iter_NN]
        for li, layer_iter in enumerate(number_layer):
          latent_space= readHDF5file(RootPathLatentSpace,\
                      'LatentSpace_Model%s_Loss%s_Launch%d_Layer%d,hdf5'%(name_NN_list[iter_NN],\
                                                                        name_loss_list[iter_loss],\
                                                                        launch_num + 1,\
                                                                        layer_iter),\
                         ['latent_space'])[0]
          if iter_NN == 0:
            compress_list = umap_shape_mnist
          else:
            compress_list = umap_shape_unet
          
          for dim_iter, dim in enumerate(compress_list):
            if dim != 0:
              ls_pca = pca_result(latent_space, dim)
              f1_score_pca = TrainPerceptron(ls_pca)
              logg.info('%d / %d, %d / %d, %d / %d, %d / %d, %d / %d, %d / %d pca score = %f'%(iter_NN + 1, len(name_NN_list), \
                                     iter_loss + 1, len(name_loss_list),\
                                     launch_num + 1, 5,\
                                     data_iter + 1, len(NamesDataSet),\
                                     li + 1, len(number_layer),\
                                     dim_iter + 1, len(compress_list),\
                                     f1_score_pca))
              precision[iter_NN, iter_loss, launch_num, data_iter, li,\
                        dim_iter, 0] = f1_score_pca
              ls_umap = umap_result(latent_space, dim)
              f1_score_umap = TrainPerceptron(ls_umap)
              logg.info('%d / %d, %d / %d, %d / %d, %d / %d, %d / %d, %d / %d umap score = %f'%(iter_NN + 1, len(name_NN_list), \
                                     iter_loss + 1, len(name_loss_list),\
                                     launch_num + 1, 5,\
                                     data_iter + 1, len(NamesDataSet),\
                                     li + 1, len(number_layer),\
                                     dim_iter + 1, len(compress_list),\
                                     f1_score_umap))

              precision[iter_NN, iter_loss, launch_num, data_iter, li,\
                        dim_iter, 1] = f1_score_umap

            else:
              f1_score = TrainPerceptron(latent_space)
              logg.info('%d / %d, %d / %d, %d / %d, %d / %d, %d / %d, %d / %d score = %f'%(iter_NN + 1, len(name_NN_list), \
                                     iter_loss + 1, len(name_loss_list),\
                                     launch_num + 1, 5,\
                                     data_iter + 1, len(NamesDataSet),\
                                     li + 1, len(number_layer),\
                                     dim_iter + 1, len(compress_list),\
                                     f1_score))

              precision[iter_NN, iter_loss, launch_num, data_iter, li,\
                        dim_iter, 0] = f1_score
            
            ff = h5.File(fs.join(RootPathLatentSpace, 'preceptron', 'perceptron.hdf5'), 'w')
            ff.create_dataset('precision', precision)
            ff.close()
#%%
"""
NN1 
"""
RootPathLatentSpace = '/data/vsevolod/DmiDiplom/LatentSpace2T/preceptron'

logg.basicConfig(filename=fs.join(RootPathLatentSpace, "LatentSpaceLogger.log"), level=logg.INFO)
logg

umap_shape_mnist = [0, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]

PathToDataSet = '/data/vsevolod/DmiDiplom/DataSets'
PathToModel = '/data/vsevolod/DmiDiplom/Models/gridNN'

NamesDataSet = ['OnlyColor.hdf5',\
                'OnlyH.hdf5',\
                'OnlyX.hdf5',\
                'Only.hdf5']
      
name_loss_list = ['weighted_categorical_crossentropy',\
             'dice_loss']
  
name_NN_list = ['ezConvAutoEncoderForMnist', 'DymasUnetСircumcised',\
             'DymasUnetWithSeparableConvСircumcised']

num_layers = [6, 7]

iter_NN = 0
for iter_loss in range(len(name_loss_list)):
  for launch_num in range(5):        
    for data_iter, data in enumerate(NamesDataSet):
      number_layer = num_layers[iter_NN]
      for li, layer_iter in enumerate(number_layer):
        latent_space= readHDF5file(RootPathLatentSpace,\
                      'LatentSpace_Model%s_Loss%s_Launch%d_Layer%d,hdf5'%(name_NN_list[iter_NN],\
                                                                        name_loss_list[iter_loss],\
                                                                        launch_num + 1,\
                                                                        layer_iter),\
                         ['latent_space'])[0]
        if iter_NN == 0:
          compress_list = umap_shape_mnist
        else:
          compress_list = umap_shape_unet
          
        precision = np.ones(len(compress_list), 2)
          
        for dim_iter, dim in enumerate(compress_list):
          if dim != 0:
            ls_pca = pca_result(latent_space, dim)
            f1_score_pca = TrainPerceptron(ls_pca)
            logg.info('%d / %d, %d / %d, %d / %d, %d / %d, %d / %d, %d / %d pca score = %f'%(iter_NN + 1, len(name_NN_list), \
                                     iter_loss + 1, len(name_loss_list),\
                                     launch_num + 1, 5,\
                                     data_iter + 1, len(NamesDataSet),\
                                     li + 1, len(number_layer),\
                                     dim_iter + 1, len(compress_list),\
                                     f1_score_pca))
            precision[dim_iter, 0] = f1_score_pca
            ls_umap = umap_result(latent_space, dim)
            f1_score_umap = TrainPerceptron(ls_umap)
            logg.info('%d / %d, %d / %d, %d / %d, %d / %d, %d / %d, %d / %d umap score = %f'%(iter_NN + 1, len(name_NN_list), \
                                     iter_loss + 1, len(name_loss_list),\
                                     launch_num + 1, 5,\
                                     data_iter + 1, len(NamesDataSet),\
                                     li + 1, len(number_layer),\
                                     dim_iter + 1, len(compress_list),\
                                     f1_score_umap))
            precision[dim_iter, 1] = f1_score_umap

          else:
            f1_score = TrainPerceptron(latent_space)
            logg.info('%d / %d, %d / %d, %d / %d, %d / %d, %d / %d, %d / %d score = %f'%(iter_NN + 1, len(name_NN_list), \
                                     iter_loss + 1, len(name_loss_list),\
                                     launch_num + 1, 5,\
                                     data_iter + 1, len(NamesDataSet),\
                                     li + 1, len(number_layer),\
                                     dim_iter + 1, len(compress_list),\
                                     f1_score))
            precision[dim_iter, 0] = f1_score
            precision[dim_iter, 1] = f1_score

            
          ff = h5.File(fs.join(RootPathLatentSpace, 'preceptron',\
              'perceptron_Model%s_Loss%s_Launch%d_Layer%d.hdf5'%(name_NN_list[0],\
                                                                  name_loss_list[iter_loss],\
                                                                  launch_num + 1,\
                                                                  layer_iter)), 'w')
          ff.create_dataset('precision', precision)
          ff.close()
#%%
"""
NN2 
"""
RootPathLatentSpace = '/data/vsevolod/DmiDiplom/LatentSpace2T/preceptron'

logg.basicConfig(filename=fs.join(RootPathLatentSpace, "LatentSpaceLogger.log"), level=logg.INFO)
logg

umap_shape_mnist = [0, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]

PathToDataSet = '/data/vsevolod/DmiDiplom/DataSets'
PathToModel = '/data/vsevolod/DmiDiplom/Models/gridNN'

NamesDataSet = ['OnlyColor.hdf5',\
                'OnlyH.hdf5',\
                'OnlyX.hdf5',\
                'Only.hdf5']
      
name_loss_list = ['weighted_categorical_crossentropy',\
             'dice_loss']
  
name_NN_list = ['ezConvAutoEncoderForMnist', 'DymasUnetСircumcised',\
             'DymasUnetWithSeparableConvСircumcised']

num_layers = [6, 7]

iter_NN = 0
for iter_loss in range(len(name_loss_list)):
  for launch_num in range(5):        
    for data_iter, data in enumerate(NamesDataSet):
      number_layer = num_layers[iter_NN]
      for li, layer_iter in enumerate(number_layer):
        latent_space= readHDF5file(RootPathLatentSpace,\
                      'LatentSpace_Model%s_Loss%s_Launch%d_Layer%d,hdf5'%(name_NN_list[iter_NN],\
                                                                        name_loss_list[iter_loss],\
                                                                        launch_num + 1,\
                                                                        layer_iter),\
                         ['latent_space'])[0]
        if iter_NN == 0:
          compress_list = umap_shape_mnist
        else:
          compress_list = umap_shape_unet
          
        precision = np.ones(len(compress_list), 2)
          
        for dim_iter, dim in enumerate(compress_list):
          if dim != 0:
            ls_pca = pca_result(latent_space, dim)
            f1_score_pca = TrainPerceptron(ls_pca)
            logg.info('%d / %d, %d / %d, %d / %d, %d / %d, %d / %d, %d / %d pca score = %f'%(iter_NN + 1, len(name_NN_list), \
                                     iter_loss + 1, len(name_loss_list),\
                                     launch_num + 1, 5,\
                                     data_iter + 1, len(NamesDataSet),\
                                     li + 1, len(number_layer),\
                                     dim_iter + 1, len(compress_list),\
                                     f1_score_pca))
            precision[dim_iter, 0] = f1_score_pca
            ls_umap = umap_result(latent_space, dim)
            f1_score_umap = TrainPerceptron(ls_umap)
            logg.info('%d / %d, %d / %d, %d / %d, %d / %d, %d / %d, %d / %d umap score = %f'%(iter_NN + 1, len(name_NN_list), \
                                     iter_loss + 1, len(name_loss_list),\
                                     launch_num + 1, 5,\
                                     data_iter + 1, len(NamesDataSet),\
                                     li + 1, len(number_layer),\
                                     dim_iter + 1, len(compress_list),\
                                     f1_score_umap))
            precision[dim_iter, 1] = f1_score_umap

          else:
            f1_score = TrainPerceptron(latent_space)
            logg.info('%d / %d, %d / %d, %d / %d, %d / %d, %d / %d, %d / %d score = %f'%(iter_NN + 1, len(name_NN_list), \
                                     iter_loss + 1, len(name_loss_list),\
                                     launch_num + 1, 5,\
                                     data_iter + 1, len(NamesDataSet),\
                                     li + 1, len(number_layer),\
                                     dim_iter + 1, len(compress_list),\
                                     f1_score))
            precision[dim_iter, 0] = f1_score
            precision[dim_iter, 1] = f1_score

            
          ff = h5.File(fs.join(RootPathLatentSpace, 'preceptron',\
              'perceptron_Model%s_Loss%s_Launch%d_Layer%d.hdf5'%(name_NN_list[0],\
                                                                  name_loss_list[iter_loss],\
                                                                  launch_num + 1,\
                                                                  layer_iter)), 'w')
          ff.create_dataset('precision', precision)
          ff.close()


NamesDataSet = ['OnlyColor.hdf5',\
                'OnlyH.hdf5',\
                'OnlyX.hdf5',\
                'Only.hdf5']
  
i = 0
print(NamesDataSet[i][:-5])
