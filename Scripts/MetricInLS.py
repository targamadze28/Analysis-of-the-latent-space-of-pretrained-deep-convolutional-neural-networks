#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 00:22:16 2020

@author: vsevolod
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

import os.path as fs
from numba import njit, float32, int32, prange
import networkx as nx
from scipy.spatial import Delaunay
import umap
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
   raise RuntimeError('Список имен групп и длина списка данных не соответствуют!')
  
  ff = h5.File(fs.join(PathToSave, SavedFileName), 'w')
  for i, group in enumerate(list_group_name):
    ff.create_dataset(group, data = data[i])
  ff.close()
  return None
#%%
  
"""
Metrics
"""
@njit([float32[:, :](float32[:, :], float32[:, :], int32[:, :])], parallel = True)
def euclidean_numba2(x, y, indexes):
  num_samples, num_feat = x.shape
  dist_matrix = np.zeros((num_samples, num_samples), dtype = np.float32)
  for i in prange(len(indexes)):
    for j in indexes[i]:
      for k in indexes[i]:
        dist_matrix[j][k] = np.sqrt(((x[j] - y[k])**2).sum())
  return dist_matrix

@njit([float32[:, :](float32[:, :], float32[:, :])], parallel = True)
def euclidean_numba2(x, y):
  num_samples, num_feat = x.shape
  dist_matrix = np.zeros((num_samples, num_samples))
  for i in range(num_samples):
    for j in numba.prange(num_samples):
      dist_matrix[i][j] = ((x[i] - y[j])**2).sum()
  return dist_matrix


def umap_result(activations, n_comp):
    embedding = umap.UMAP(n_components=n_comp).fit_transform(activations)
    return embedding
  
def check_symmetric(a, tol=1e-5):
    return np.all(np.abs(a-a.T) < tol)
  
def rectangle_distance(parameters_1, parameters_2):
  kappa = 170
  xsi = 64
  S = 0
  for i in range(3):
    S += (1./kappa)*np.abs(parameters_1[i] - parameters_2[i])
  for i in range(3, 7):
    S += (1./xsi)*np.abs(parameters_1[i] - parameters_2[i])
  return S

#%%
RootPathToLatentSpace = '/home/vsevolod/Desktop/Dymas/LatentSpace'
    
NamesDataSet = ['FreeColor.hdf5',\
                'FreeColorAndH.hdf5',\
                'FreeColorAndX.hdf5',\
                'FreeColorAndHL.hdf5',\
                'FreeColorAndXY.hdf5',\
                'FreeAll.hdf5']

PathToModel = '/home/vsevolod/Desktop/Dymas/Models'
NameModel = ['model_ezConvAutoEncoderForMnist_dice_loss.hdf5',\
             'model_DymasUnetСircumcised_dice_loss.hdf5',\
              'model_DymasUnetWithSeparableConvСircumcised_dice_loss.hdf5']

for j, data in enumerate(NamesDataSet):
  NameFile = 'latent_space_%s_%s'%(NameModel[0], data)
  latent_space = readHDF5file(RootPathToLatentSpace, NameFile, ['latent_space'])[0]
  latent_space = umap_result(latent_space, 16)
  tri = Delaunay(latent_space)
  dist_matrix = euclidean_numba2(latent_space, latent_space, tri.simplices)
  ff = h5.File(fs.join(RootPathToLatentSpace, 'DistanceMatrix',\
                     'dm_%s_%s'%(NameModel[0], data)), 'w')
  ff.create_dataset('distance', data = dist_matrix)
  ff.close()
  
#%%
    
PathToDataSet = '/home/vsevolod/Desktop/Dymas/DataSets'
  
ff = h5.File(fs.join(RootPathToLatentSpace, 'DistanceMatrix',\
                     'dm_%s_%s'%(NameModel[0], NamesDataSet[2])), 'r')
dist_matrix = ff['distance'][...]
ff.close()

G = nx.from_numpy_matrix(dist_matrix)

center = nx.center(G)
perm = np.random.permutation(len(center))
center = center[perm[0]]

print(center)

fileData = readHDF5file(PathToDataSet, NamesDataSet[1],\
                            ['parameters', 'image'])
parameters_all = (fileData[0].astype(np.float32))[:5000]
images = (fileData[1][:5000])

dist = np.empty((5000,), dtype = np.float32)
for k in range(5000):
  dist_tmp = nx.dijkstra_path(G, center, k)
  print(k)
  S = 0
  for i in range(len(dist_tmp) - 1):
    S += dist_matrix[dist_tmp[i], dist_tmp[i + 1]]
  dist[k] = S

ff = h5.File(fs.join(RootPathToLatentSpace, 'DistanceMatrix',\
                     'dist_on_graph_%s_%s'%(NameModel[0], NamesDataSet[1])), 'w')
ff.create_dataset('distance', data = dist)
ff.close()


dist_rectangle = np.empty((5000,), dtype = np.float32)

for i in range(5000):
  dist_rectangle[i] = rectangle_distance(parameters_all[center, 0, :], parameters_all[i, 0, :])


print(dist_rectangle.shape)

plt.scatter(dist_rectangle, dist,\
                    c = 'blue')
plt.show()


dist = nx.dijkstra_path(G, center, 32)

S = 0
for i in range(len(dist) - 1):
  S += dist_matrix[dist[i], dist[i + 1]]
  

path = dict(nx.all_pairs_shortest_path_length(G))
for i in range(100):
  print(path[0][i])
