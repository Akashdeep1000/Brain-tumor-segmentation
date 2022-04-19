# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:57:02 2022

@author: Akashdeep
"""

from tensorflow.keras.models import load_model
import numpy as np
import os
from a2_custom_datagen import imageLoader
from sklearn.model_selection import KFold
from monai.metrics import compute_meandice
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import torch
import sys

print("start")
sys.stdout.flush()
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

print("before model load")
sys.stdout.flush()

mymodel_1 = load_model("/home/cap6412.student38/0_brats_a2.hdf5", compile = False)
mymodel_2 = load_model("/home/cap6412.student38/1_brats_a2.hdf5", compile = False)
mymodel_3 = load_model("/home/cap6412.student38/2_brats_a2.hdf5", compile = False)
mymodel_4 = load_model("/home/cap6412.student38/3_brats_a2.hdf5", compile = False)
mymodel_5 = load_model("/home/cap6412.student38/4_brats_a2.hdf5", compile = False)

dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

Hausdorff_metric_batch = HausdorffDistanceMetric(include_background=True, reduction="mean_batch")

print("after model load")
sys.stdout.flush()

batch_size= 4


kf = KFold(n_splits=5)
for (train, test), model in zip(kf.split(all_img_list), [mymodel_1,mymodel_2,mymodel_3,mymodel_4,mymodel_5]):
    print("first loop")
    sys.stdout.flush()
    
    steps_per_epoch = len(test)//batch_size
    i = 0
    
    test_img_datagen = imageLoader(val_img_dir, all_img_list, val_mask_dir, all_mask_list, batch_size, test, test)

    
    for test_image_batch, test_mask_batch in test_img_datagen:
    
      
      test_mask_batch = test_mask_batch.transpose(0,4,2,3,1)
  
      test_mask_batch = torch.round(torch.tensor(test_mask_batch))
  
      test_pred_batch = model.predict(test_image_batch)
  
      test_pred_batch = test_pred_batch.transpose(0,4,2,3,1)
  
      test_pred_batch = torch.round(torch.tensor(test_pred_batch))
      
      dice_metric_batch(y_pred=test_pred_batch, y=test_mask_batch)

      i+=1
      print(str(i) + '/' + str(steps_per_epoch))
      sys.stdout.flush()
      if i == steps_per_epoch:
        print("break")
        break;
      
    metric_batch = dice_metric_batch.aggregate()
    metric_tc = metric_batch[0].item()
    metric_wt = metric_batch[1].item()
    metric_et = metric_batch[2].item()
    dice_metric_batch.reset()
    print("Dice score")
    sys.stdout.flush()
    print(metric_tc, metric_wt, metric_et)
    sys.stdout.flush()


kf = KFold(n_splits=5)
for (train, test), model in zip(kf.split(all_img_list), [mymodel_1,mymodel_2,mymodel_3,mymodel_4,mymodel_5]):
    
    test_img_datagen = imageLoader(val_img_dir, all_img_list, val_mask_dir, all_mask_list, batch_size, test, test)
    
    steps_per_epoch = len(test)//batch_size
    i = 0
    
    for test_image_batch, test_mask_batch in test_img_datagen:
    
        
      test_mask_batch = test_mask_batch.transpose(0,4,2,3,1)
  
      test_mask_batch = torch.round(torch.tensor(test_mask_batch))
  
      test_pred_batch = model.predict(test_image_batch)
  
      test_pred_batch = test_pred_batch.transpose(0,4,2,3,1)
  
      test_pred_batch = torch.round(torch.tensor(test_pred_batch))
      
      Hausdorff_metric_batch(y_pred=test_pred_batch, y=test_mask_batch)
      
      i+=1
      print(str(i) + '/' + str(steps_per_epoch))
      sys.stdout.flush()
      if i == steps_per_epoch:
        print("break")
        break;
      
    metric_batch = Hausdorff_metric_batch.aggregate()
    metric_tc = metric_batch[0].item()
    metric_wt = metric_batch[1].item()
    metric_et = metric_batch[2].item()
    
    print("hausdroff distance") 
    sys.stdout.flush()
    print(metric_tc, metric_wt, metric_et)
    sys.stdout.flush()
