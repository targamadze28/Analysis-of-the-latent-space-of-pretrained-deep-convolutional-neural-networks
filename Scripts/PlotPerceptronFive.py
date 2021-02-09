#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import os.path as fs
#%%
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
   raise RuntimeError('Group name list and data list length do not match!')
  
  ff = h5.File(fs.join(PathToSave, SavedFileName), 'w')
  for i, group in enumerate(list_group_name):
    ff.create_dataset(group, data = data[i])
  ff.close()
  return None
#%%
def plotDependenceAccuracy4Compress(dims, data, title = '', saveplot = True,\
                                    Path2Save = '', NamePic = ''):
  plt.figure(figsize = (16, 10))
  plt.plot(dims, data[:, 0], label = 'class 1', \
           color = 'b')
  plt.scatter(dims, data[:, 0], \
           color = 'b')
  
  plt.plot(dims, data[:, 1], label = 'class 2',\
           color = 'y')
  plt.scatter(dims, data[:, 1],\
              color = 'y')
  
  plt.plot(dims, data[:, 2], label = 'class 3',\
           color = 'g')
  plt.scatter(dims, data[:, 2],\
           color = 'g')
  
  plt.xlabel('log dimension')
  plt.ylabel('accuracy')
  plt.ylim([0, 1.2])
  plt.xscale('log')
  plt.title(title)
  plt.legend()
  if saveplot:
    plt.show()
  else: 
    plt.savefig(fs.join(Path2Save, NamePic))
  return None
#%%
RootPath = ''
Path2PerceptronData = fs.join(RootPath, '')
NameFile = ''

data = readHDF5file(Path2PerceptronData, NameFile, ['precision'])[0]

umap_shape_mnist = []

plt.figure(figsize = (16, 10))
plt.plot(umap_shape_mnist, data[:, 0, 0], label = 'class 1', \
         color = 'b')
plt.scatter(umap_shape_mnist, data[:, 0, 0], \
         color = 'b')

plt.plot(umap_shape_mnist, data[:, 0, 1], label = 'class 2',\
         color = 'y')
plt.scatter(umap_shape_mnist, data[:, 0, 1],\
            color = 'y')

plt.plot(umap_shape_mnist, data[:, 0, 2], label = 'class 3',\
         color = 'g')
plt.scatter(umap_shape_mnist, data[:, 0, 2],\
         color = 'g')

plt.xlabel('log dimension')
plt.ylabel('accuracy')
plt.ylim([0, 1.2])
plt.xscale('log')
plt.legend()
plt.show()

plotDependenceAccuracy4Compress(umap_shape_mnist, data[:, 1, :])

#%%
"""
NN1 ezConvAutoEncoder4Mnist
"""
RootPath = ''
Path2PerceptronData = fs.join(RootPath, 'perceptron')

NamesDataSet = ['']
      
name_loss_list = ['']
  
name_NN_list = ['']

num_layers = []
umap_shape_mnist = []

iter_NN = 0
for iter_loss in range(1):#range(len(name_loss_list)):
  iter_loss = 1
  for layer_iter in num_layers:
    for dataiter, dataname in enumerate(NamesDataSet):
      for launch_num in range(5):        
        NameFile = 'perceptron_Model%s_Loss%s_Launch%d_Data%s_Layer%d.hdf5'%(name_NN_list[0],\
                                                                  name_loss_list[iter_loss],\
                                                                  launch_num + 1,\
                                                                  dataname[:-5],\
                                                                  layer_iter)
        data = readHDF5file(Path2PerceptronData, NameFile, ['precision'])[0]
        title = 'PCA NN %d, Loss dice, Launch %d, Data №%d, layer %d'%(iter_NN,\
                                                                       launch_num + 1,\
                                                                       dataiter + 1,\
                                                                        layer_iter)
        plotDependenceAccuracy4Compress(umap_shape_mnist, data[:, 0, :], title,\
                                        saveplot = False, Path2Save = fs.join(RootPath, 'perceptron png'),\
                                        NamePic = title + '.png')
        title = 'Umap NN %d, Loss dice, Launch %d, Data №%d, layer %d'%(iter_NN,\
                                                                       launch_num + 1,\
                                                                       dataiter + 1,\
                                                                        layer_iter)
        plotDependenceAccuracy4Compress(umap_shape_mnist, data[:, 1, :], title,\
                                        saveplot = False, Path2Save = fs.join(RootPath, 'perceptron png'),\
                                        NamePic = title + '.png')
          
#%%
"""
NN2 ezConvAutoEncoder4Mnist
"""
RootPath = ''
Path2PerceptronData = fs.join(RootPath, '')

NamesDataSet = ['']
      
name_loss_list = ['']
  
name_NN_list = ['']

num_layers = []
umap_shape_mnist = []

iter_NN = 1
for iter_loss in range(1):#range(len(name_loss_list)):
  iter_loss = 1
  for layer_iter in num_layers:
    for dataiter, dataname in enumerate(NamesDataSet):
      for launch_num in range(5):        
        NameFile = 'perceptron_Model%s_Loss%s_Launch%d_Data%s_Layer%d.hdf5'%(name_NN_list[0],\
                                                                  name_loss_list[iter_loss],\
                                                                  launch_num + 1,\
                                                                  dataname[:-5],\
                                                                  layer_iter)
        data = readHDF5file(Path2PerceptronData, NameFile, ['precision'])[0]
        title = 'PCA NN %d, Loss dice, Launch %d, Data №%d, layer %d'%(iter_NN,\
                                                                       launch_num + 1,\
                                                                       dataiter + 1,\
                                                                        layer_iter)
        plotDependenceAccuracy4Compress(umap_shape_mnist, data[:, 0, :], title,\
                                        saveplot = False, Path2Save = fs.join(RootPath, 'perceptron2 png'),\
                                        NamePic = title + '.png')
        title = 'Umap NN %d, Loss dice, Launch %d, Data №%d, layer %d'%(iter_NN,\
                                                                       launch_num + 1,\
                                                                       dataiter + 1,\
                                                                        layer_iter)
        plotDependenceAccuracy4Compress(umap_shape_mnist, data[:, 1, :], title,\
                                        saveplot = False, Path2Save = fs.join(RootPath, 'perceptron2 png'),\
                                        NamePic = title + '.png')
