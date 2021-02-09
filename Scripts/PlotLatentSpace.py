#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5 
import os.path as fs
import umap

from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE

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

def umap_result(activations, n_comp):
  embedding = umap.UMAP(n_components=n_comp).fit_transform(activations)
  return embedding

def tsne_result(activations, n_comp, n_jobs = 8):
  embedding = TSNE(n_components = n_comp, n_jobs = n_jobs, perplexity = 100).fit_transform(activations)
  return embedding

def pca_result(activations, n_comp, n_jobs = 32):
  embedding = PCA(n_components= n_comp).fit_transform(activations)
  return embedding


#%%
"""
Plot Latent Space
"""  
from sklearn.decomposition import PCA  

RootPathToLatentSpace = ''
    
NamesDataSet = ['']
PathToModel = ''
NameModel = ['']
NNName = ['']
DataName = ['']
#Umap2D
for i, model in enumerate(NameModel):
  NameFile = 'latent_space_%s_%s'%(model, NamesDataSet[-1])
  #latent_space = readHDF5file(RootPathToLatentSpace, NameFile, ['latent_space'])[0]
  #um = umap.UMAP(n_components=16).fit(latent_space)

  for j, data in enumerate(NamesDataSet):
    print(model, data)

    NameFile = 'latent_space_%s_%s'%(model, data)
    latent_space = readHDF5file(RootPathToLatentSpace, NameFile, ['latent_space'])[0]
    #latent_space_8d = um.transform(latent_space)
    
    #latent_space_2d = umap_result(latent_space_8d, 2)
    latent_space_2d = pca_result(latent_space, 2)
    
    plt.figure(figsize=(16,10))
    plt.scatter(latent_space_2d[:5000, 0], latent_space_2d[:5000, 1], s=4, alpha=0.7, label='first class')
    plt.scatter(latent_space_2d[5000:10000, 0],latent_space_2d[5000:10000, 1], s=4, alpha=0.7, label='second class')
    plt.scatter(latent_space_2d[10000:, 0],latent_space_2d[10000:, 1], s=4, alpha=0.7, label='third class')
    plt.title('%s'%(NameFile))
    plt.savefig(fs.join(RootPathToLatentSpace,'pictures_umaplast','%s%spca.png'%(NNName[i], DataName[j]))) 

"""
#Umap3D
for i, model in enumerate(NameModel):
  for data in NamesDataSet:
    print(model, data)

    NameFile = 'latent_space_%s_%s'%(model, data)
    latent_space = readHDF5file(RootPathToLatentSpace, NameFile, ['latent_space'])[0]
    print(latent_space.shape)
    #latent_space_3d = umap_result(latent_space, 2)
    #saveHDF5file(fs.join(RootPathToLatentSpace, 'Unet2D'), '%s_%s.hdf5'%(model[:-5], data[:-5]), ['ls_umap'], [latent_space_3d])
"""
#%%
RootPathToLatentSpace = ''
    
NamesDataSet = ['']
PathToModel = ''
NameModel = ['']

NNName = ['']
DataName = ['']

for i, model in enumerate(NameModel):
  NameFile = 'latent_space_%s_%s'%(model, NamesDataSet[-1])
  #latent_space = readHDF5file(RootPathToLatentSpace, NameFile, ['latent_space'])[0]
  #um = umap.UMAP(n_components=16).fit(latent_space)

  for j, data in enumerate(NamesDataSet):
    print(model, data)

    NameFile = 'latent_space_%s_%s'%(model, data)
    latent_space = readHDF5file(RootPathToLatentSpace, NameFile, ['latent_space'])[0]
    #latent_space_8d = um.transform(latent_space)
    
    latent_space_2d = umap_result(latent_space, 2)
    #latent_space_2d = pca_result(latent_space, 2)
    
    plt.figure(figsize=(16,10))
    plt.scatter(latent_space_2d[:15000, 0], latent_space_2d[:15000, 1], s=4, alpha=0.7, label='first class')
    plt.scatter(latent_space_2d[15000:30000, 0],latent_space_2d[15000:30000, 1], s=4, alpha=0.7, label='second class')
    plt.scatter(latent_space_2d[30000:, 0],latent_space_2d[30000:, 1], s=4, alpha=0.7, label='third class')
    plt.title('%s'%(NameFile))
    plt.savefig(fs.join(RootPathToLatentSpace,'umap2d','%s%sumap2d.png'%(NNName[i], DataName[j])))
