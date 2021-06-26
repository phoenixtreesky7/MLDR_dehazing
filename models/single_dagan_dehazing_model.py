import itertools
import sys

import torch
import torch.nn.functional as F
from util.image_pool import ImagePool
import math

from models import hdr_networks
from epdn import epdn_networks
from .base_model import BaseModel
import numpy as np
from models.gradient import gradient
sys.path.append('D:/dzhao/dehazing_360/SSDH-HDR')
from epdn.models import create_pretrained_model
from models.gradient import gradient
from util.mindloss import MINDLoss
from ECLoss.ECLoss import DCLoss
from TVLoss.TVLossL1 import TVLossL1 

class SUDH_HDR(BaseModel):
	def name(self):
		return 'SUDH_HDR'
	
	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		self.opt = opt
		self.gpu_ids
		self.device = "cuda" 
		self.dehazing_loss = opt.dehazing_loss
		self.opt.batchSize = opt.batchSize
		self.unet_layer = opt.unet_layer
		# specify the training losses you want to print out. The program will call base_model.get_current_losses
		self.loss_names = ['idt_rec_CC', 'idt_rec_CHC', 'idt_rec_HH', 'idt_rec_HCH', 'idt_CC', 'idt_HC', 'idt_CS', 'idt_HS', 'G_C', 'G_H']

		if opt.gradient_loss:
			self.loss_names.extend(['gradient_fake_CH', 'gradient_fake_HC', 'gradient_rec_CC', 'gradient_rec_HH', 'gradient_rec_CHC', 'gradient_rec_HCH'])

		if opt.gradient_dehaze_loss:
			self.loss_names.extend(['gradient_dehaze_fake_CH', 'gradient_dehaze_real_H'])
		self.loss_names.extend(['G_dehaze_fake_CH','G_dehaze_real_H', 'idt_dehaze_fake_CH','idt_dehaze_real_H', 'G_VGG_dehaze_fake_CH', 'G_VGG_dehaze_real_H', 'DCP_dehaze_fake_CH', 'TV_dehaze_fake_CH', 'DCP_dehaze_real_H', 'TV_dehaze_real_H'])


		# specify the images you want to save/display. The program will call base_model.get_current_visuals
		visual_names_A_1 = ['real_C', 'fake_CH', 'rec_CHC', 'rec_CC', 'dehaze_fake_CH']
		visual_names_B_1 = ['real_H', 'fake_HC', 'rec_HCH', 'rec_HH', 'dehaze_real_H']
		self.visual_names = visual_names_A_1 + visual_names_B_1
		if opt.dehazing_loss:
			visual_names_C_1 = ['dehaze_real_H', 'dehaze_fake_CH']


		if opt.gradient_loss:
			visual_names_G = ['gradient_fake_CH', 'gradient_fake_HC', 'gradient_rec_CC', 'gradient_rec_HH', 'gradient_rec_CHC', 'gradient_rec_HCH', 'gradient_dehaze_real_H', 'gradient_dehaze_fake_CH']
			self.visual_names += visual_names_G
			print('visual_names ----------', self.visual_names)




		# specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
		if self.isTrain:
			if opt.dehazing_loss:
				self.model_names = ['EC_C', 'EC_H', 'DC_C', 'DC_H', 'G_D', 'D_C', 'D_H', 'D_D_C', 'D_D_H']
			else:
				self.model_names = ['EC_C', 'EC_H', 'DC_C', 'DC_H', 'D_C', 'D_H']
		
		else:  # during test time, only load Gs
			self.model_names = ['EC_C', 'EC_H', 'DC_C', 'DC_H', 'G_D']
		
		# load/define networks
		# The naming conversion is different from those used in the paper
		# Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
		self.netEC_C = hdr_networks.define_Encoder(opt.input_nc, opt.output_nc,
		                                  opt.ngf, opt.n_width, opt.n_height, opt.n_blocks_local, opt.which_model_netG, opt.norm,
		                                  not opt.no_dropout, opt.init_type, self.gpu_ids)

		self.netDC_C = hdr_networks.define_Decoder(opt.input_nc, opt.output_nc,
		                                  opt.ngf, opt.n_width, opt.n_height, opt.n_blocks_local, opt.which_model_netG, opt.norm,
		                                  not opt.no_dropout, opt.init_type, self.gpu_ids)

		self.netEC_H = hdr_networks.define_Encoder(opt.output_nc, opt.input_nc,
			                                  opt.ngf, opt.n_width, opt.n_height, opt.n_blocks_local, opt.which_model_netG, opt.norm,
			                                  not opt.no_dropout, opt.init_type, self.gpu_ids)
		self.netDC_H = hdr_networks.define_Decoder(opt.output_nc, opt.input_nc,
			                                  opt.ngf, opt.n_width, opt.n_height, opt.n_blocks_local, opt.which_model_netG, opt.norm,
			                                  not opt.no_dropout, opt.init_type, self.gpu_ids)

		if opt.dehazing_loss:
			self.netG_D = epdn_networks.define_G(opt.input_nc, opt.output_nc, opt.epdn_ngf, opt.dehazing_netG,
			                                     opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
			                                     opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

		if self.isTrain:
			use_sigmoid = opt.no_lsgan
			# netD_A : source domain discriminator
			self.netD_C = hdr_networks.define_D(opt.output_nc, opt.ndf,
			                                      opt.which_model_netD,
			                                      opt.n_layers_D, opt.norm, use_sigmoid,
			                                      opt.init_type, self.gpu_ids)

			# netD_B : target domain discriminator
			self.netD_H = hdr_networks.define_D(opt.input_nc, opt.ndf,
			                                      opt.which_model_netD,
			                                      opt.n_layers_D, opt.norm, use_sigmoid,
			                                      opt.init_type, self.gpu_ids)

			# netD_D : dehazing discriminator
			if opt.dehazing_loss:
				self.netD_D_C = hdr_networks.define_D(opt.input_nc, opt.ndf,
				                                  opt.which_model_netD,
				                                  opt.n_layers_D, opt.norm, use_sigmoid,
				                                  opt.init_type, self.gpu_ids)
				self.netD_D_H = hdr_networks.define_D(opt.input_nc, opt.ndf,
				                                  opt.which_model_netD,
				                                  opt.n_layers_D, opt.norm, use_sigmoid,
				                                  opt.init_type, self.gpu_ids)

		if self.isTrain:
			self.fake_CH_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
			self.fake_HC_pool = ImagePool(opt.pool_size)
			self.fake_CHC_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
			self.fake_HCH_pool = ImagePool(opt.pool_size)
			if opt.dehazing_loss:
				self.dehaze_fake_CH_pool = ImagePool(opt.pool_size)
				self.dehaze_real_H_pool = ImagePool(opt.pool_size)
			# define loss functions
			self.criterionGAN = hdr_networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
			#self.criterionEPDNGAN = epdn_networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
			self.criterionIdt = torch.nn.L1Loss()
			self.criterionMSE = torch.nn.MSELoss()
			#self.criterionTV = torch.nn.TVLoss()
			self.criterionVGG = epdn_networks.VGGLoss(self.gpu_ids)
			self.criterionMIND = MINDLoss()
			self.criterionFeat = torch.nn.L1Loss()
			#self.criterionDC = DCLoss(self.opt)
			#self.criterionTV = TVLossL1()   netD_D
			# initialize optimizers
			self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_C.parameters(), self.netD_H.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
			
			
			self.optimizer_G = torch.optim.Adam(itertools.chain(self.netEC_C.parameters(), self.netEC_H.parameters(), self.netDC_C.parameters(), self.netDC_H.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

			if opt.dehazing_loss:
				self.optimizer_G_D = torch.optim.Adam(self.netG_D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
				self.optimizer_D_D_C = torch.optim.Adam(self.netD_D_C.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
				self.optimizer_D_D_H = torch.optim.Adam(self.netD_D_H.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

			self.optimizers = []
			self.optimizers.append(self.optimizer_G)
			self.optimizers.append(self.optimizer_D)

			if opt.dehazing_loss:
				self.optimizers.append(self.optimizer_G_D)
				self.optimizers.append(self.optimizer_D_D_C)
				self.optimizers.append(self.optimizer_D_D_H)

	def set_input(self, input):
		self.real_C = input['A'].to(self.device)
		self.real_H = input['B'].to(self.device)
		
		self.image_paths_A = input['A_paths']
		self.image_paths_B = input['B_paths']
		self.image_paths = self.image_paths_A + self.image_paths_B
	
	def forward(self, opt):
		#！！ C->H; H->C ！！#
		# Disentanglement#
		# CF: clear image feature;  CC: clear image content;  CS: clear image style.
		# HF: hazy               ;  HC: hazy               ;  HS: hazy              
		self.CF, self.CC, self.CS = self.netEC_C(self.real_C)
		self.HF, self.HC, self.HS = self.netEC_H(self.real_H)

		# Transference #
		self.fake_CH = self.netDC_H(self.CF, self.CC, self.HS)
		self.fake_HC = self.netDC_C(self.HF, self.HC, self.CS)

		# Short Reconstruction #
		self.rec_CC = self.netDC_C(self.CF, self.CC, self.CS)
		self.rec_HH = self.netDC_H(self.HF, self.HC, self.HS)

		#！！ C->H->C; H->C->H ！！#
		# Disentanglement #
		self.fake_CHF, self.fake_CHC, self.fake_CHS = self.netEC_H(self.fake_CH)
		self.fake_HCF, self.fake_HCC, self.fake_HCS = self.netEC_C(self.fake_HC)

		# Transference #
		self.rec_CHC = self.netDC_C(self.fake_CHF, self.fake_CHC, self.fake_HCS)
		self.rec_HCH = self.netDC_H(self.fake_HCF, self.fake_HCC, self.fake_CHS)

		if opt.dehazing_loss:
			#!! Dehazing !!#
			_, self.dehaze_fake_CH = self.netG_D(self.fake_CH)
			_, self.dehaze_real_H = self.netG_D(self.real_H)

#======================================================================
#======================================================================

	def backward_D_basic(self, netD, real, fake, DT=False):
		# Real
		if DT == True:
			pred_real = netD(real.detach())
		else:
			pred_real = netD(real)
		loss_D_real = self.criterionGAN(pred_real, True)
		# Fake
		pred_fake = netD(fake.detach())
		loss_D_fake = self.criterionGAN(pred_fake, False)
		# Combined loss
		loss_D = (loss_D_real + loss_D_fake) * 0.5
		# backward
		
		loss_D.backward()
		return loss_D

	def backward_D(self):
		if not self.opt.imagepool:
			fake_CH = self.fake_CH_pool.query(self.fake_CH)
			fake_HC = self.fake_HC_pool.query(self.fake_HC)
			rec_CHC = self.fake_CHC_pool.query(self.rec_CHC)
			rec_HCH = self.fake_HCH_pool.query(self.rec_HCH)
		else:
			fake_CH = self.fake_CH
			fake_HC = self.fake_HC
			rec_CHC = self.rec_CHC
			rec_HCH = self.rec_HCH

		# GAN: fake_CH is hazy? fake_HC is clear?
		self.loss_D_CH = self.backward_D_basic(self.netD_H, self.real_H, fake_CH) 
		self.loss_D_HC = self.backward_D_basic(self.netD_C, self.real_C, fake_HC)

		self.loss_D_CHC = self.backward_D_basic(self.netD_C, self.real_C, rec_CHC) 
		self.loss_D_HCH = self.backward_D_basic(self.netD_H, self.real_H, rec_HCH)


	def backward_D_D_C(self):
		dehaze_fake_CH = self.dehaze_fake_CH_pool.query(self.dehaze_fake_CH)
		self.loss_D_D_C = self.backward_D_basic(self.netD_D_C, self.real_C, dehaze_fake_CH)

	def backward_D_D_H(self):
		dehaze_real_H = self.dehaze_real_H_pool.query(self.dehaze_real_H)
		self.loss_D_D_H = self.backward_D_basic(self.netD_D_H, self.fake_HC, dehaze_real_H, DT=True)

	def backward_G(self, opt):
		lambda_idt = self.opt.lambda_identity
		lambda_C = self.opt.lambda_content
		lambda_S = self.opt.lambda_style

		# Identity loss
		if lambda_idt > 0:
			#self.idt_A_1 = self.netG_A_1(self.real_B)
			self.loss_idt_rec_CC = self.criterionMSE(self.rec_CC, self.real_C) * lambda_idt
			self.loss_idt_rec_CHC = self.criterionMSE(self.rec_CHC, self.real_C) * lambda_idt
			#self.idt_A_2 = self.netG_A_2(self.real_B)
			self.loss_idt_rec_HH = self.criterionMSE(self.rec_HH, self.real_H) * lambda_idt
			self.loss_idt_rec_HCH = self.criterionMSE(self.rec_HCH, self.real_H) * lambda_idt

			#self.loss_idt_dehaze_fake_CH = self.criterionMSE(self.dehaze_fake_CH, self.real_C) * lambda_idt
			#self.loss_idt_dehaze_real_H = self.criterionMSE(self.dehaze_real_H, self.fake_HC) * lambda_idt

			self.loss_idt_CC = 0
			self.loss_idt_CS = 0
			self.loss_idt_HC = 0
			self.loss_idt_HS = 0
			esum = 0
			for i in range(self.unet_layer):
				esum += math.exp(i)
			#print('esum', esum)
			for i in range(self.unet_layer):
				#print('CC', self.CC[i].shape, 'fake_CHC', self.fake_CHC[i].shape, 'HC', self.HC[i].shape, 'fake_HCH', self.fake_HCC[i].shape)
				#print('(i/self.unet_layer)', self.unet_layer)
				self.loss_idt_CC += self.criterionMSE(self.CC[i], self.fake_CHC[i]) * lambda_C * (math.exp(i)/esum)
				self.loss_idt_CS += self.criterionIdt(self.CS[i], self.fake_HCS[i]) * lambda_S * (math.exp(i)/esum)
				self.loss_idt_HC += self.criterionMSE(self.HC[i], self.fake_HCC[i]) * lambda_C * (math.exp(i)/esum)
				self.loss_idt_HS += self.criterionIdt(self.HS[i], self.fake_CHS[i]) * lambda_S * (math.exp(i)/esum)

			self.loss_G = self.loss_idt_rec_CC + self.loss_idt_rec_CHC + self.loss_idt_rec_HH + self.loss_idt_rec_HCH + self.loss_idt_CC + self.loss_idt_CS + self.loss_idt_HC + self.loss_idt_HS
		else:
			print('#### NOTE!!! opt.lambda_identity should > 0 !!!! ###')

		self.loss_G_C = 0.5 * self.criterionGAN(self.netD_C(self.fake_HC), True)
		self.loss_G_H = 0.5 * self.criterionGAN(self.netD_H(self.fake_CH), True)
		self.loss_G += self.loss_G_C + self.loss_G_H
		#
		if opt.gradient_loss:
			self.gradient_real_C = gradient(self.real_C)
			self.gradient_real_H = gradient(self.real_H)
			self.gradient_fake_CH = gradient(self.fake_CH)
			self.gradient_fake_HC = gradient(self.fake_HC)
			
			self.gradient_rec_HH = gradient(self.rec_HH)
			self.gradient_rec_CC = gradient(self.rec_CC)
			self.gradient_rec_HCH = gradient(self.rec_HCH)
			self.gradient_rec_CHC = gradient(self.rec_CHC)

			if opt.dehazing_loss:
				self.gradient_dehaze_real_H = gradient(self.dehaze_real_H)
				self.gradient_dehaze_fake_CH = gradient(self.dehaze_fake_CH)

			self.loss_gradient_fake_CH = 0.01 * self.criterionIdt(self.gradient_real_C, self.gradient_fake_CH)
			self.loss_gradient_fake_HC = 0.01 * self.criterionIdt(self.gradient_real_H, self.gradient_fake_HC)
			self.loss_gradient_rec_CC = 0.01 * self.criterionIdt(self.gradient_real_C, self.gradient_rec_CC)
			self.loss_gradient_rec_HH = 0.01 * self.criterionIdt(self.gradient_real_H, self.gradient_rec_HH)
			self.loss_gradient_rec_CHC = 0.01 * self.criterionIdt(self.gradient_real_C, self.gradient_rec_CHC)
			self.loss_gradient_rec_HCH = 0.01 * self.criterionIdt(self.gradient_real_H, self.gradient_rec_HCH)
			self.loss_G += self.loss_gradient_fake_CH + self.loss_gradient_fake_HC + self.loss_gradient_rec_CC + self.loss_gradient_rec_HH + self.loss_gradient_rec_CHC + self.loss_gradient_rec_HCH
		else:
			self.loss_gradient_fake_CH = 0
			self.loss_gradient_fake_HC = 0
			self.loss_gradient_rec_CC = 0
			self.loss_gradient_rec_HH = 0
			self.loss_gradient_rec_CHC = 0
			self.loss_gradient_rec_HCH = 0

			

		if opt.dehazing_loss:
			if opt.DHN_frozen_epoch != -1 and opt.current_epoch > opt.DHN_frozen_epoch:
				# GAN loss (Fake Passability Loss)
				#dehaze_fake_B_1_temp = self.dehaze_fake_B_1_pool.query(self.dehaze_fake_B_1)
				pred_fake = self.netD_D_C(self.dehaze_fake_CH)
				self.loss_G_dehaze_fake_CH = self.criterionGAN(pred_fake, True)
				pred_fake2 = self.netD_D_H(self.dehaze_real_H)
				self.loss_G_dehaze_real_H = self.criterionGAN(pred_fake2, True)

				self.loss_G_D = self.loss_G_dehaze_fake_CH + self.loss_G_dehaze_real_H

				# gradient loss
				if opt.gradient_dehaze_loss:
					

					self.loss_gradient_dehaze_fake_CH = self.criterionIdt(self.gradient_real_C, self.gradient_dehaze_fake_CH)
					self.loss_gradient_dehaze_real_H = self.criterionIdt(self.gradient_fake_HC, self.gradient_dehaze_real_H)
					self.loss_G_D += self.loss_gradient_dehaze_fake_CH + self.loss_gradient_dehaze_real_H
				else:
					self.loss_gradient_dehaze_fake_CH = 0
					self.loss_gradient_dehaze_real_H = 0

				# reconstruction loss
				self.loss_idt_dehaze_fake_CH = opt.dynamic_weight * self.criterionMSE(self.real_C, self.dehaze_fake_CH)
				self.loss_idt_dehaze_real_H = opt.dynamic_weight * self.criterionMSE(self.fake_HC, self.dehaze_real_H)

				self.loss_G_D += self.loss_idt_dehaze_fake_CH + self.loss_idt_dehaze_real_H


				# VGG feature matching loss
				self.loss_G_VGG_dehaze_fake_CH = self.criterionVGG(self.dehaze_fake_CH, self.real_C) * self.opt.lambda_vgg
				self.loss_G_VGG_dehaze_real_H = self.criterionVGG(self.dehaze_real_H, self.fake_HC) * self.opt.lambda_vgg
				self.loss_G_D += self.loss_G_VGG_dehaze_fake_CH + self.loss_G_VGG_dehaze_real_H


				# TV loss and DCP loss
				self.loss_DCP_dehaze_fake_CH = opt.lambda_DC * DCLoss((self.dehaze_fake_CH+1)/2, self.opt)
				self.loss_TV_dehaze_fake_CH = opt.lambda_TV * TVLossL1(self.dehaze_fake_CH)
				self.loss_DCP_dehaze_real_H = opt.lambda_DC * DCLoss((self.dehaze_real_H+1)/2, self.opt)
				self.loss_TV_dehaze_real_H = opt.lambda_TV * TVLossL1(self.dehaze_real_H)

				self.loss_G_D += self.loss_DCP_dehaze_fake_CH + self.loss_TV_dehaze_fake_CH + self.loss_DCP_dehaze_real_H + self.loss_TV_dehaze_real_H


				self.loss_G += self.loss_G_D

			else:
				self.loss_G_dehaze_fake_CH = 0
				self.loss_G_dehaze_real_H = 0
				self.loss_gradient_dehaze_fake_CH = 0
				self.loss_gradient_dehaze_real_H = 0
				self.loss_idt_dehaze_fake_CH = 0
				self.loss_idt_dehaze_real_H = 0
				self.loss_G_VGG_dehaze_fake_CH = 0
				self.loss_G_VGG_dehaze_real_H = 0
				self.loss_DCP_dehaze_fake_CH = 0
				self.loss_TV_dehaze_fake_CH = 0
				self.loss_DCP_dehaze_real_H = 0
				self.loss_TV_dehaze_real_H = 0
				self.loss_G_D = 0


		
		self.loss_G.backward()


	def optimize_parameters(self, opt):
		# forward
		self.forward(opt)
		# G_A and G_B
		# set D to false, back prop G's gradients
		#if opt.Shared_DT:
		#	self.set_requires_grad([self.netD_A, self.netD_B_1, self.netD_B_2], False)
		#else:
		self.set_requires_grad([self.netD_C, self.netD_H], False)
		if opt.dehazing_loss:
			self.set_requires_grad([self.netD_D_C], False)
			self.set_requires_grad([self.netD_D_H], False)

		self.set_requires_grad([self.netEC_C, self.netEC_C, self.netDC_H, self.netDC_H], True)
		if opt.dehazing_loss:
			self.set_requires_grad([self.netG_D], True)
		
		self.optimizer_G.zero_grad()
		if opt.dehazing_loss:
			self.optimizer_G_D.zero_grad()
		# self.optimizer_CLS.zero_grad()
		self.backward_G(opt)
		self.optimizer_G.step()
		if opt.dehazing_loss:
			self.optimizer_G_D.step()

		# D_A and D_B
		self.set_requires_grad([self.netD_C, self.netD_H], True)
		self.optimizer_D.zero_grad()
		self.backward_D()
		self.optimizer_D.step()

		if opt.dehazing_loss:
			self.set_requires_grad([self.netD_D_C], True)
			self.optimizer_D_D_C.zero_grad()
			self.backward_D_D_C()
			self.optimizer_D_D_C.step()

			self.set_requires_grad([self.netD_D_H], True)
			self.optimizer_D.zero_grad()
			self.optimizer_D_D_H.zero_grad()
			self.backward_D_D_H()
			self.optimizer_D_D_H.step()



