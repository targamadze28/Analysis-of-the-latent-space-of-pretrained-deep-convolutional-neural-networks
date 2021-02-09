#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import cv2 as cv 
import os.path as fs

#%%
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

def readHDF5file(PathToSave, SavedFileName, list_group_name):
  data = []
  ff = h5.File(fs.join(PathToSave, SavedFileName), 'r')
  for group in list_group_name:
    data.append(ff[group][...])
  ff.close()
  return data

#%%
def selection_bias(x_center, y_center, l, h):
  selection = True
  if x_center + l > 64:
    selection = False
  if x_center - l < 0:
    selection = False
  if y_center + h > 64:
    selection = False
  if y_center - h < 0:
    selection = False
  return selection
#%%
""" 
The input data is a 64x64x3 RGB image containing three colored rectangles.
the rectangle has three parameters center, size, color. Classes do not overlap in color ranges.
The output data of segmentation are semantic masks, which are binary images 64 x 64. The number of recognized classes is equal to 4:three colored rectangular objects and a background class.
"""
def create_dataset(num_samples, images_size, delta, perm = []):
  images = np.zeros((num_samples, images_size, images_size, 3), dtype = np.uint8)
  mask = np.zeros((num_samples, images_size, images_size, 3), dtype = np.uint8)
  mask_background = np.zeros((num_samples, images_size, images_size, 3), dtype = np.uint8)
  full_mask = np.zeros((num_samples, images_size, images_size, 4), dtype = np.uint8)
  parameters = np.zeros((num_samples, 3, 7), dtype = np.int32)
      
  for sample in range(num_samples):
    
    if sample % 1000 == 0:
      print(sample, ' / ', num_samples)
    
    perm = np.random.permutation(3)
    
    center_1 = 42
    center_2 = 127 
    center_3 = 212
    
    for rec_class in perm:
      #selection = False
      #x_center, y_center = np.random.randint(0, 64), np.random.randint(0, 64)
      #l, h = np.random.randint(0, 64), np.random.randint(0, 64)

      #x_center, y_center = np.random.randint(0, 64), np.random.randint(0, 64)
      #l, h = np.random.randint(0, 64), np.random.randint(0, 64)
      
      x_center, y_center = 32, 32
      l, h = 10, 10

      #while selection == False:
      #  x_center, y_center = np.random.randint(0, 64), np.random.randint(0, 64)
      #  l, h = np.random.randint(0, 64), np.random.randint(0, 64)
      #  selection = selection_bias(x_center, y_center, l, h)
          
      
      if rec_class + 1 == 1:
        Color1 = np.random.randint(center_1 - delta, center_1 + delta)
        Color2 = np.random.randint(center_2 - delta, center_2 + delta)
        Color3 = np.random.randint(center_3 - delta, center_3 + delta)
      elif rec_class + 1 == 2:
        Color1 = np.random.randint(center_2 - delta, center_2 + delta)
        Color2 = np.random.randint(center_3 - delta, center_3 + delta)
        Color3 = np.random.randint(center_1 - delta, center_1 + delta)
      elif rec_class + 1 == 3:
        Color1 = np.random.randint(center_3 - delta, center_3 + delta)
        Color2 = np.random.randint(center_1 - delta, center_1 + delta)
        Color3 = np.random.randint(center_2 - delta, center_2 + delta)
      
      """
      if rec_class + 1 == 1:
        Color1 = center_1
        Color2 = center_2
        Color3 = center_3
      elif rec_class + 1 == 2:
        Color1 = center_2 
        Color2 = center_3
        Color3 = center_1
      elif rec_class + 1 == 3:
        Color1 = center_3 
        Color2 = center_1
        Color3 = center_2
      """
        
        
      parameters[sample, rec_class, :3] = np.asarray([Color1, Color2, Color3])
      parameters[sample, rec_class, 3:5] = np.asarray([l, h])
      parameters[sample, rec_class, 5:7] = np.asarray([x_center, y_center])
      
      color = (Color1, Color2, Color3)
      images[sample] = cv.rectangle(images[sample], (x_center - int(l/2), y_center + int(h/2)),\
                                     (x_center + int(l/2), y_center - int(h/2)),\
                                       color, -1) 
      color_mask = [0, 0, 0]
      color_mask[rec_class] = 1
      mask[sample] = cv.rectangle(mask[sample], \
                                    (x_center - int(l/2), y_center + int(h/2)),\
                                     (x_center + int(l/2), y_center - int(h/2)),\
                                       tuple(color_mask), -1) 
        
      mask_background[sample] = cv.rectangle(mask_background[sample], \
                                    (x_center - int(l/2), y_center + int(h/2)),\
                                     (x_center + int(l/2), y_center - int(h/2)),\
                                       (1, 0, 0), -1)
      
      image_tmp = np.zeros((images_size, images_size), dtype = np.uint8)
      image_tmp = cv.rectangle(image_tmp,\
                               (x_center - int(l/2), y_center + int(h/2)),
                               (x_center + int(l/2), y_center - int(h/2)),\
                                       1, -1) 


      full_mask[sample, :, :, rec_class] = image_tmp
    
    full_mask[sample, :, :, -1] = mask_background[sample, :, :, 0]
        
  return images, np.concatenate((mask, mask_background[:, :, :, :1]), axis = -1), parameters, full_mask
  
PathToSave = ''
NameDataSetFile = ''

num_samples = 100
images_size = 64
delta = 15 # lenght color range

images, mask, parameters, full_mask = create_dataset(num_samples, images_size, delta, [0])
#images_1, mask_1, parameters_1, full_mask_1 = create_dataset(num_samples, images_size, delta, [0])
#images_2, mask_2, parameters_2, full_mask_2 = create_dataset(num_samples, images_size, delta, [1])
#images_3, mask_3, parameters_3, full_mask_3 = create_dataset(num_samples, images_size, delta, [2])

#images = np.concatenate((images_1, images_2, images_3))
#mask = np.concatenate((mask_1, mask_2, mask_1))
#parameters = np.concatenate((parameters_1, parameters_2, parameters_3))
#full_mask = np.concatenate((full_mask_1, full_mask_2, full_mask_3))


mask[:, :, :, -1] = 1 - mask[:, :, :, -1]
full_mask[:, :, :, -1] = 1 - full_mask[:, :, :, -1]



saveHDF5file(PathToSave, NameDataSetFile, ['image', 'mask', 'parameters', 'full_mask'],\
             [images, mask, parameters, full_mask])

#data = readHDF5file(PathToSave, NameDataSetFile, ['image', 'mask']
