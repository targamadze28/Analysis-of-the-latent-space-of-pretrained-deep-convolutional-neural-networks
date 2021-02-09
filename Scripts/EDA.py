#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:42:35 2020

@author: vsevolod
"""

import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

import os.path as fs

def readHDF5file(PathToSave, SavedFileName, list_group_name):
  data = []
  ff = h5.File(fs.join(PathToSave, SavedFileName), 'r')
  for group in list_group_name:
    data.append(ff[group][...])
  ff.close()
  return data


def find_square(mask):
  squeare = np.zeros((4, ), dtype = np.float32)
  n = len(mask)
  for sample in range(n):
    for class_rect in range(4):
      squeare[class_rect] += np.count_nonzero(mask[sample, :, :, class_rect])
  return squeare

#%%
"""
Exploration Data Analysis
"""
RootPathtoFile = '/home/vsevolod/Desktop/Dymas/DataSets'
NameFileWithoutSelection = 'EDA_DatasetWithoutSelection.hdf5'
NameFileWithSelection = 'EDA_DatasetWithSelection.hdf5'


fileDatawos = readHDF5file(RootPathtoFile, NameFileWithoutSelection,\
                        ['image', 'mask', 'parameters', 'full_mask'])


fileDataws = readHDF5file(RootPathtoFile, NameFileWithSelection,\
                        ['image', 'mask', 'parameters', 'full_mask'])

imageswos = fileDatawos[0]
maskwos = fileDatawos[1]
paramswos = fileDatawos[2]
full_maskwos = fileDatawos[3]

imagesws = fileDataws[0]
maskws = fileDataws[1]
paramsws = fileDataws[2]
full_maskws = fileDataws[3]

#%%
plt.hist(np.reshape(paramsws[:, :, 5], (len(paramsws)*3, 1)), bins = 64,\
         facecolor='b', alpha=0.75)
plt.title('центр по оси x, набор данных с отбором')
plt.xlabel('положение на картинке 0-64')
plt.ylabel('количество изображений')
plt.show()

plt.hist(np.reshape(paramswos[:, :, 5], (len(paramsws)*3, 1)), bins = 64,\
         facecolor='g', alpha=0.75)
plt.title('центр по оси x, набор данных без отбора')
plt.xlabel('положение на картинке 0-64')
plt.ylabel('количество изображений')
plt.show()

plt.hist(np.reshape(paramsws[:, :, 3], (len(paramsws)*3, 1)), bins = 64,\
         facecolor='b', alpha=0.75)
plt.title('длина прямоугольника, набор данных с отбором')
plt.xlabel('положение на картинке 0-64')
plt.ylabel('количество изображений')
plt.show()

plt.hist(np.reshape(paramswos[:, :, 3], (len(paramsws)*3, 1)), bins = 64,\
         facecolor='g', alpha=0.75)
plt.title('длина прямоугольника, набор данных без отбора')
plt.xlabel('положение на картинке 0-64')
plt.ylabel('количество изображений')
plt.show()

#%%
Swos = find_square(maskwos)
print(Swos/(50000*64*64))

Swosfull = find_square(full_maskwos)
print(Swosfull/(50000*64*64))

print((Swosfull - Swos)/Swosfull)

Sws = find_square(maskws)
print(Sws/(50000*64*64))

Swsfull = find_square(full_maskws)
print(Swsfull/(50000*64*64))

print((Swsfull - Sws)/Swsfull)
