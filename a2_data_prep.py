# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 00:56:13 2022

@author: Akashdeep
"""
import numpy as np
import nibabel as nib
import glob
#import matplotlib.pyplot as plt
import splitfolders
from sklearn.preprocessing import MinMaxScaler
from monai.transforms import Transform

scaler = MinMaxScaler()

#dataset loading
class ConvertToMultiChannelBasedOnBratsClassesd(Transform):
	def __call__(self, data):
		
	
		result = []
		result.append(np.logical_or(data == 2, data == 3))
		result.append(
	                np.logical_or(
	                    np.logical_or(data == 2, data == 3), data == 1
	                )
	            )
		result.append(data == 2)
		data = np.stack(result, axis=0).astype(np.float32)
		
		return data
	

            
	   

image_list = sorted(glob.glob("/home/cap6412.student38/Task01_BrainTumor/imagesTr/*.nii.gz"))
mask_list = sorted(glob.glob("/home/cap6412.student38/Task01_BrainTumor/labelsTr/*.nii.gz"))



for img in range(len(image_list)):
	print("Now preparing image and masks number: ", img)
	temp_image=nib.load(image_list[img]).get_fdata()
	temp_image=scaler.fit_transform(temp_image.reshape(-1, temp_image.shape[-1])).reshape(temp_image.shape)
	
	temp_mask=nib.load(mask_list[img]).get_fdata()
	temp_combined_images = np.vstack([temp_image])
	
	
	temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
	temp_mask = temp_mask[56:184, 56:184, 13:141]
	
	
	temp_mask = ConvertToMultiChannelBasedOnBratsClassesd()(temp_mask)

	temp_mask = temp_mask.transpose(1, 2, 3, 0)
	temp_mask = temp_mask.astype(np.float32)	
	
	val, counts = np.unique(temp_mask, return_counts= True)
	
 
	if (1 - (counts[0]/counts.sum())) > 0.01:
		print("Save Me")		
		np.save('/home/cap6412.student38/Task01_BrainTumor/input_data/images/image_'+str(img)+'.npy', temp_combined_images)
		np.save('/home/cap6412.student38/Task01_BrainTumor/input_data/masks/mask_'+str(img)+'.npy', temp_mask)
		
	else:
		print("NULL")

		

input_folder = '/home/cap6412.student38/Task01_BrainTumor/input_data/'
output_folder = '/home/cap6412.student38/Task01_BrainTumor/input_data_1/'


splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None) # default values
