import os.path
import random

import numpy as np
from PIL import Image
from data.base_dataset import BaseDataset, get_transform
#from data.cityscapes import remap_labels_to_train_ids
from data.image_folder import make_cs_labels, make_dataset

# This dataset is used to conduct double cyclegan for both GTAV->CityScapes and Synthia->CityScapes
class DH_HDR_SSPV_MS(BaseDataset):
	def initialize(self, opt):
		# OHAZE as dataset 1
		# 3D60 as dataset 2
		
		self.opt = opt
		self.root = opt.dataroot
		self.dir_A1 = os.path.join(opt.dataroot, 'reside', 'clear')
		self.dir_B1 = os.path.join(opt.dataroot, 'reside', 'hazy')
		self.dir_A2 = os.path.join(opt.dataroot, 'ohaze', 'clear')
		self.dir_B2 = os.path.join(opt.dataroot, 'ohaze', 'hazy')
		self.dir_C = os.path.join(opt.dataroot, 'real')
		

		self.A1_paths = make_dataset(self.dir_A1)
		self.B1_paths = make_dataset(self.dir_B1)
		self.A2_paths = make_dataset(self.dir_A2)
		self.B2_paths = make_dataset(self.dir_B2)
		self.C_paths = make_dataset(self.dir_C)
		

		self.A1_paths = sorted(self.A1_paths)
		self.B1_paths = sorted(self.B1_paths)
		self.A2_paths = sorted(self.A2_paths)
		self.B2_paths = sorted(self.B2_paths)
		self.C_paths = sorted(self.C_paths)

		self.A1_size = len(self.A1_paths)
		self.A2_size = len(self.A2_paths)
		self.C1_size = len(self.C1_paths)

		self.transform = get_transform(opt)

	
	def __getitem__(self, index):
		A1_path = self.A1_paths[index % self.A1_size]
		B1_path = self.B1_paths[index % self.A1_size]
		#C_path = self.C_paths[index % self.C_size]
		
		if self.opt.serial_batches:
			index_2 = index % self.B1_size
		else:
			index_2 = random.randint(0, self.B_size - 1)

		A2_path = self.A2_paths[index_2]
		B2_path = self.B2_paths[index_2]
		
		index_C = random.randint(0, self.C_size - 1)
		C_path = self.C_paths[index_C]

		A1_img = Image.open(A1_path).convert('RGB')
		B1_img = Image.open(B1_path).convert('RGB')
		A2_img = Image.open(A2_path).convert('RGB')
		B2_img = Image.open(B2_path).convert('RGB')
		C_img = Image.open(C_path).convert('RGB')

		A1 = self.transform(A1_img)
		B1 = self.transform(B1_img)
		A2 = self.transform(A2_img)
		B2 = self.transform(B2_img)
		C = self.transform(C_img)

		

		if self.opt.which_direction == 'BtoA':
			input_nc = self.opt.output_nc
			output_nc = self.opt.input_nc
		else:
			input_nc = self.opt.input_nc
			output_nc = self.opt.output_nc
		
		if input_nc == 1:  # RGB to gray
			tmp = A1[0, ...] * 0.299 + A1[1, ...] * 0.587 + A1[2, ...] * 0.114
			A1 = tmp.unsqueeze(0)

		if output_nc == 1:  # RGB to gray
			tmp = B1[0, ...] * 0.299 + B1[1, ...] * 0.587 + B1[2, ...] * 0.114
			B1 = tmp.unsqueeze(0)

		if input_nc == 1:  # RGB to gray
			tmp = A2[0, ...] * 0.299 + A2[1, ...] * 0.587 + A2[2, ...] * 0.114
			A2 = tmp.unsqueeze(0)

		if output_nc == 1:  # RGB to gray
			tmp = B2[0, ...] * 0.299 + B2[1, ...] * 0.587 + B2[2, ...] * 0.114
			B2 = tmp.unsqueeze(0)

		if output_nc == 1:  # RGB to gray
			tmp = C[0, ...] * 0.299 + C[1, ...] * 0.587 + C[2, ...] * 0.114
			C = tmp.unsqueeze(0)
		
		return {'A1': A1, 'B1': B1, 'A2': A2, 'B2': B2, 'C': C, 'A1_paths': A1_path, 'B1_paths': B1_path,'A2_paths': A2_path, 'B2_paths': B2_path, 'C_paths': C_path}
		


	def __len__(self):
		return max(self.A_size, self.B_size, self.C_size)
	
	def name(self):
		return 'DH_HDR_SSPV'
