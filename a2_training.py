# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 00:52:43 2022

@author: Akashdeep
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
import pandas as pd
import os
from a2_custom_datagen import imageLoader
from tensorflow import keras
import glob
import random
from a2_unet import unet_a2
from monai.metrics import compute_hausdorff_distance, compute_meandice
import segmentation_models_3D as sm
from sklearn.model_selection import KFold


#training

train_img_dir = "/home/cap6412.student38/Task01_BrainTumor/input_data_1/"
train_mask_dir ="/home/cap6412.student38/Task01_BrainTumor/input_data_1/"


val_img_dir = "/home/cap6412.student38/Task01_BrainTumor/input_data_1/"
val_mask_dir = "/home/cap6412.student38/Task01_BrainTumor/input_data_1/"

train_img_list=['train/images/' + f for f in os.listdir(train_img_dir + 'train/images/')]
train_mask_list = ['train/masks/' + f for f in os.listdir(train_img_dir + 'train/masks/')]

val_img_list=['val/images/' + f for f in os.listdir(train_img_dir + 'val/images/')]
val_mask_list = ['val/masks/' + f for f in os.listdir(train_img_dir + 'val/masks/')]

all_img_list = train_img_list + val_img_list
all_mask_list = train_mask_list + val_mask_list
batch_size = 2
i =0 
kf = KFold(n_splits=5)
for train, test in kf.split(all_img_list):
  train_img_datagen = imageLoader(train_img_dir, all_img_list, 
                                train_mask_dir, all_mask_list, batch_size, train, train)


  val_img_datagen = imageLoader(val_img_dir, all_img_list, 
                                val_mask_dir, all_mask_list, batch_size, test, test)




  dice_loss = sm.losses.DiceLoss() 
  
  
  metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]
  
  LR = 0.0001
  optim = keras.optimizers.Adam(LR)
  
  #fit model
  
  steps_per_epoch = len(train_img_list)//batch_size
  val_steps_per_epoch = len(val_img_list)//batch_size
  
  
  model = unet_a2(IMG_HEIGHT=128, 
                            IMG_WIDTH=128, 
                            IMG_DEPTH=128, 
                            IMG_CHANNELS=4, 
                            num_classes=3)
  
  model.compile(optimizer = optim, loss=dice_loss, metrics=metrics)
  print(model.summary())
  
  #print(model.input_shape)
  #print(model.output_shape)
  
  history=model.fit(train_img_datagen,
            steps_per_epoch=steps_per_epoch,
            epochs=100,
            verbose=1,
            validation_data=val_img_datagen,
            validation_steps=val_steps_per_epoch,
            )
  

  model.save(str(i) + '_brats_a2.hdf5')
  i+=1

loss = history.history['loss']
print(loss)
val_loss = history.history['val_loss']
print(val_loss)
#epochs = range(1, len(loss) + 1)
#plt.plot(epochs, loss, 'y', label='Training loss')
#plt.plot(epochs, val_loss, 'r', label='Validation loss')
#plt.title('Training and validation loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.show() #google save fig

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

print(acc, val_acc)

#plt.plot(epochs, acc, 'y', label='Training accuracy')
#plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
#plt.title('Training and validation accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend()
#plt.show()


