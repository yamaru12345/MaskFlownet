import os
import re
import skimage.io
from functools import lru_cache
import struct
import numpy as np


base_dir = "/content/drive/MyDrive/SOMPO/data/mask_flow_net_2/input_data/orig/"
color_img_pre_pass = base_dir + "preFrameImg_nodist_crop_noise/"
color_img_post_pass = base_dir + "postFrameImg_nodist_crop_noise/"

flow_pass = base_dir + "flow_crop/"
mask_pass = base_dir + "occ_mask/"

split_file = base_dir + "train_val.txt"

def list_data(path = None):

	dataset = []
	train_set = []
	val_set = []

	lines = []
	with open(split_file) as f:
		lines = f.readlines()

	for i in range(len(lines)):
		
		color_img_pre_name_temp = color_img_pre_pass + "preFrameImg_" + str(i) + ".png"
		color_img_post_name_temp = color_img_post_pass + "postFrameImg_" + str(i) + ".png"
		flow_data_name_temp = flow_pass + "flow_" + str(i) + ".npy"
		mask_name_temp = mask_pass + "occ_mask_" + str(i) + ".png"

		if lines[i] == "1\n": # train
			train_set.append([color_img_pre_name_temp, color_img_post_name_temp, flow_data_name_temp, mask_name_temp])
		elif lines[i] == "2\n": # val
			val_set.append([color_img_pre_name_temp, color_img_post_name_temp, flow_data_name_temp, mask_name_temp])

	dataset = [train_set, val_set]

	return dataset

def load(fname):

	if fname.endswith('png'):
		data = skimage.io.imread(fname)
		if data.ndim < 3:
			data = 255 - np.expand_dims(data, -1)
		return data
	elif fname.endswith('npy'):
		data = np.load(fname)
		return data

if __name__ == '__main__':
	dataset = list_data()
