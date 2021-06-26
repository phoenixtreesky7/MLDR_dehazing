import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from dcn.deform_conv import ModulatedDeformConvPack2 as DCN


###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer


def get_scheduler(optimizer, opt):
	if opt.lr_policy == 'lambda':
		def lambda_rule(epoch):
			lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
			return lr_l
		
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
	else:
		return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
	return scheduler


def init_weights(net, init_type='normal', gain=0.02):
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, gain)
			init.constant_(m.bias.data, 0.0)
	
	print('initialize network with %s' % init_type)
	net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
	if len(gpu_ids) > 0:
		assert (torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)
	init_weights(net, init_type)
	return net


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

#########################################

def define_G_E_A(input_nc, output_nc, ngf, n_blocks, which_model_netG, norm='instance', use_dropout=False, init_type='normal', gpu_ids=[]):
	netG = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netG == 'resnet_9blocks':
		netG = HDRResnetEncoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
	elif which_model_netG == 'resnet_6blocks':
		netG = HDRResnetEncoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3)
	elif which_model_netG == 'unet_fusecat':
		netG = HDRUNetEncoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'unet_fusecsfm':
		netG = HDRUNetEncoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
	return init_net(netG, init_type, gpu_ids)

def define_G_D_A(input_nc, output_nc, ngf, n_blocks, which_model_netG, norm='instance', use_dropout=False, init_type='normal', gpu_ids=[]):
	netG = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netG == 'resnet_9blocks':
		netG = HDRResnetDecoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
	elif which_model_netG == 'resnet_6blocks':
		netG = HDRResnetDecoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3)
	elif which_model_netG == 'unet_fusecat':
		netG = HDRUNetDecoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'unet_fusecsfm':
		netG = HDRUNetDecoderClear(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
	return init_net(netG, init_type, gpu_ids)

########################################

def define_G_E_B(input_nc, output_nc, ngf, n_blocks, haze_layer, which_model_netG, norm='instance', use_dropout=False, init_type='normal', gpu_ids=[]):
	netG = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netG == 'resnet_9blocks':
		netG = HDRResnetEncoderHazy(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
	elif which_model_netG == 'resnet_6blocks':
		netG = HDRResnetEncoderHazy(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3)
	elif which_model_netG == 'unet_fusecat':
		netG = HDRUnetEncoderHazy(input_nc, output_nc, ngf, haze_layer, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'unet_fusecsfm':
		netG = HDRUnetEncoderHazy(input_nc, output_nc, ngf, haze_layer, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
	return init_net(netG, init_type, gpu_ids)

def define_G_D_B(input_nc, output_nc, ngf, n_blocks, haze_layer, which_model_netG, fuse_model, norm='instance', use_dropout=False, init_type='normal', gpu_ids=[]):
	netG = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netG == 'resnet_9blocks':
		netG = HDRResnetDecoderHazy(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
	elif which_model_netG == 'resnet_6blocks':
		netG = HDRResnetDecoderHazy(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3)
	elif which_model_netG == 'unet_fusecat':
		netG = HDRUnetDecoderHazy(input_nc, output_nc, ngf, haze_layer, fuse_model=fuse_model, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	elif which_model_netG == 'unet_fusecsfm':
		netG = HDRUnetDecoderHazy(input_nc, output_nc, ngf, haze_layer, fuse_model=fuse_model, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, padding_type='reflect')
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
	return init_net(netG, init_type, gpu_ids)

##############################################

def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
	netD = None
	norm_layer = get_norm_layer(norm_type=norm)
	
	if which_model_netD == 'basic':
		netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
	elif which_model_netD == 'n_layers':
		netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
	elif which_model_netD == 'pixel':
		netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
	else:
		raise NotImplementedError('Discriminator model name [%s] is not recognized' %
		                          which_model_netD)
	return init_net(netD, init_type, gpu_ids)




class GANLoss(nn.Module):
	def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
		super(GANLoss, self).__init__()
		self.register_buffer('real_label', torch.tensor(target_real_label))
		self.register_buffer('fake_label', torch.tensor(target_fake_label))
		if use_lsgan:
			self.loss = nn.MSELoss()
		else:
			self.loss = nn.BCELoss()

	def get_target_tensor(self, input, target_is_real):
		if target_is_real:
			target_tensor = self.real_label
		else:
			target_tensor = self.fake_label
		return target_tensor.expand_as(input)
	
	def __call__(self, input, target_is_real):
		target_tensor = self.get_target_tensor(input, target_is_real)
		return self.loss(input, target_tensor)


class SFT_layer(nn.Module):
    def __init__(self, c_nc, h_nc):
        super(SFT_layer, self).__init__()
        Relu = nn.LeakyReLU(0.2, True)

        condition_conv1 = nn.Conv2d(h_nc, c_nc, kernel_size=3, stride=1, padding=1)
        condition_conv2 = nn.Conv2d(c_nc, c_nc, kernel_size=3, stride=1, padding=1)
        condition_conv3 = nn.Conv2d(c_nc, c_nc, kernel_size=3, stride=1, padding=1)

        conditon_conv = [condition_conv1, Relu, condition_conv2, Relu, condition_conv3, Relu]
        self.condition_conv = nn.Sequential(*conditon_conv)

        scale_conv1 = nn.Conv2d(c_nc, c_nc, kernel_size=3, stride=1, padding=1)
        scale_conv2 = nn.Conv2d(c_nc, c_nc, kernel_size=3, stride=1, padding=1)
        scale_conv = [scale_conv1, Relu, scale_conv2, Relu]
        self.scale_conv = nn.Sequential(*scale_conv)

        sift_conv1 = nn.Conv2d(c_nc, c_nc, kernel_size=3, stride=1, padding=1)
        sift_conv2 = nn.Conv2d(c_nc, c_nc, kernel_size=3, stride=1, padding=1)
        sift_conv = [sift_conv1, Relu, sift_conv2, Relu]
        self.sift_conv = nn.Sequential(*sift_conv)

    def forward(self, c, h):
        h_condition = self.condition_conv(h)
        scaled_feature = self.scale_conv(h_condition) * c
        sifted_feature = scaled_feature + self.sift_conv(h_condition)

        return sifted_feature

class RSABlock(nn.Module):

    def __init__(self, input_channel=32, output_channel=32, offset_channel=32):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        if self.in_channel != self.out_channel:
            self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        self.dcnpack = DCN(output_channel, output_channel, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                            extra_offset_mask=True, offset_in_channel=offset_channel)
        self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x, offset):
        if self.in_channel != self.out_channel:
            x = self.conv0(x)
        fea = self.lrelu(self.dcnpack([x, offset]))
        out = self.conv1(fea) + x
        return out
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

class OffsetBlock(nn.Module):

    def __init__(self, input_channel=32, offset_channel=32, last_offset=False):
        super().__init__()
        self.offset_conv1 = nn.Conv2d(input_channel, offset_channel, 3, 1, 1)  # concat for diff
        if last_offset:
            self.offset_conv2 = nn.Conv2d(offset_channel*2, offset_channel, 3, 1, 1)  # concat for offset
        self.offset_conv3 = nn.Conv2d(offset_channel, offset_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x, last_offset=None):
        offset = self.lrelu(self.offset_conv1(x))
        if last_offset is not None:
            last_offset = F.interpolate(last_offset, scale_factor=2, mode='bilinear', align_corners=False)
            offset = self.lrelu(self.offset_conv2(torch.cat([offset, last_offset * 2], dim=1)))
        offset = self.lrelu(self.offset_conv3(offset))
        return offset
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

########################################################################
########################################################################



########################################################################
########################################################################
# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class HDRUNetEncoderClear(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(HDRUNetEncoderClear, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		
		encode_contfe = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		self.encode_contfe = nn.Sequential(*encode_contfe)

		mult = 1
		encode_cont1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		self.encode_cont1 = nn.Sequential(*encode_cont1)

		mult = 2
		encode_cont2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		self.encode_cont2 = nn.Sequential(*encode_cont2)

		mult = 4
		encode_cont3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		self.encode_cont3 = nn.Sequential(*encode_cont3)

		

	def forward(self, input):
		ecfe = self.encode_contfe(input)
		ec1 = self.encode_cont1(ecfe)
		ec2 = self.encode_cont2(ec1)
		ec3 = self.encode_cont3(ec2)
		

		return [ecfe, ec1, ec2, ec3]


##########################################################
class HDRUNetDecoderClear(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(HDRUNetDecoderClear, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		n_downsampling = 3
		mult = 2 ** n_downsampling
		decode_ft = []
		for i in range(n_blocks):
			decode_ft += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		self.decode_ft = nn.Sequential(*decode_ft)

		offset_channel = ngf
		self.offset_ft = OffsetBlock(ngf * mult, offset_channel, False)
		self.dres_ft = RSABlock(ngf * mult, ngf * mult, offset_channel)

		## 3rd decoder  ##
		mult = 2 ** (n_downsampling+1)
		decoder_up3 = [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
		decoder_up3 += [nn.ConvTranspose2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]
		self.decoder_up3 = nn.Sequential(*decoder_up3)

		self.offset_3 = OffsetBlock(int(ngf * mult / 4), offset_channel, False)
		self.dres_3 = RSABlock(int(ngf * mult / 4), int(ngf * mult / 4), offset_channel)

		##  2nd decoder  ##
		mult = 2 ** (n_downsampling)
		decoder_up2 = [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
		decoder_up2 += [nn.ConvTranspose2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]

		self.decoder_up2 = nn.Sequential(*decoder_up2)

		self.offset_2 = OffsetBlock(int(ngf * mult / 4), offset_channel, False)
		self.dres_2 = RSABlock(int(ngf * mult / 4), int(ngf * mult / 4), offset_channel)

		##  1st decoder  ##
		mult = 2 ** (n_downsampling-1)
		decoder_up1 = [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
		decoder_up1 += [nn.ConvTranspose2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]

		self.decoder_up1 = nn.Sequential(*decoder_up1)

		self.offset_1 = OffsetBlock(int(ngf * mult / 4), offset_channel, False)
		self.dres_1 = RSABlock(int(ngf * mult / 4), int(ngf * mult / 4), offset_channel)

		##   final decoder   ##
		mult = 2 ** (n_downsampling-2)
		decoder_upfn = [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
		self.decoder_upfn = nn.Sequential(*decoder_upfn)

		self.offset_fn = OffsetBlock(int(ngf * mult / 4), offset_channel, False)
		self.dres_fn = RSABlock(int(ngf * mult / 4), int(ngf * mult / 4), offset_channel)

		decoder_fn = [nn.ReflectionPad2d(3)]
		decoder_fn += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
		decoder_fn += [nn.Tanh()]
		
		self.decoder_fn = nn.Sequential(*decoder_fn)
	
	def forward(self, input, c):
		dft = self.decode_ft(c[3])
		ofs_ft = self.offset_ft(dft, None)
		rsa_ft = self.dres_ft(dft, ofs_ft)

		concat = torch.cat((c[3], rsa_ft), 1)
		dup3 = self.decoder_up3(concat)
		ofs_3 = self.offset_3(dup3, ofs_ft)
		rsa_3 = self.dres_3(dup3, ofs_3)

		concat = torch.cat((c[2], dup3), 1)
		dup2 = self.decoder_up2(concat)
		ofs_2 = self.offset_2(dup2, ofs_3)
		rsa_2 = self.dres_2(dup2, ofs_2)

		concat = torch.cat((c[1], dup2), 1)
		dup1 = self.decoder_up1(concat)
		ofs_1 = self.offset_1(dup1, ofs_2)
		rsa_1 = self.dres_1(dup1, ofs_1)

		concat = torch.cat((c[0], dup1), 1)
		dupfn = self.decoder_upfn(concat)
		ofs_fn = self.offset_fn(dupfn, ofs_1)
		rsa_fn = self.dres_fn(dupfn, ofs_fn)

		out = self.decoder_upfn(rsa_fn) + input

		return out

##############################################################

########################################################################
# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class HDRUnetEncoderHazy(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, haze_layer=2, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(HDRUnetEncoderHazy, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		self.hl = haze_layer
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		## Content Encoder ##
		encode_contfe = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		self.encode_contfe = nn.Sequential(*encode_contfe)

		mult = 1
		encode_cont1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		self.encode_cont1 = nn.Sequential(*encode_cont1)

		mult = 2
		encode_cont2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		self.encode_cont2 = nn.Sequential(*encode_cont2)

		mult = 4
		encode_cont3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		self.encode_cont3 = nn.Sequential(*encode_cont3)

		

		## Haze Encoder ##
		encode_hazefe = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.LeakyReLU(0.2, True)]
		self.encode_hazefe = nn.Sequential(*encode_hazefe)

		# hazy style layer 1 #
		mult = 1
		encode_ds1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
				                    stride=2, padding=1, bias=use_bias),
				          norm_layer(ngf * mult * 2),
				          nn.LeakyReLU(0.2, True)]
		self.encode_ds1 = nn.Sequential(*encode_ds1)
		if self.hl == 3:
			n_blocks = 2
			encode_haze1 = []
			for i in range(n_blocks):
				encode_haze1 += [nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3,
				                    stride=1, padding=1, bias=use_bias),
				          norm_layer(ngf * mult * 2),
				          nn.LeakyReLU(0.2, True)]
			self.encode_haze1 = nn.Sequential(*encode_haze1)

		# hazy style layer 2 #
		mult = 2
		encode_ds2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
				                    stride=2, padding=1, bias=use_bias),
				          norm_layer(ngf * mult * 2),
				          nn.LeakyReLU(0.2, True)]
		self.encode_ds2 = nn.Sequential(*encode_ds2)

		if self.hl == 2 or self.hl == 3:
			encode_haze2 = []
			for i in range(n_blocks):
				encode_haze2 += [nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3,
				                    stride=1, padding=1, bias=use_bias),
				          norm_layer(ngf * mult * 2),
				          nn.LeakyReLU(0.2, True)]
			self.encode_haze2 = nn.Sequential(*encode_haze2)

		# hazy style layer 3 #
		mult = 4
		encode_ds3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		self.encode_ds3 = nn.Sequential(*encode_ds3)

		encode_haze3 = []
		for i in range(n_blocks):
			encode_haze3 += [nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3,
			                    stride=1, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.LeakyReLU(0.2, True)]
		self.encode_haze3 = nn.Sequential(*encode_haze3)

		# hazy style layer f #
		#n_downsampling = 3
		#mult = 2 ** n_downsampling
		#encode_hazeft = []
		#for i in range(n_blocks):
		#	encode_hazeft += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3,
		#	                    stride=1, padding=1, bias=use_bias),
		#	          norm_layer(ngf * mult),
		#	          nn.LeakyReLU(0.2, True)]
		#self.encode_hazeft = nn.Sequential(*encode_hazeft)

	def forward(self, input):
		ecfe = self.encode_contfe(input)
		ec1 = self.encode_cont1(ecfe)
		ec2 = self.encode_cont2(ec1)
		ec3 = self.encode_cont3(ec2)

		ehfe = self.encode_hazefe(input)
		eds1 = self.encode_ds1(ehfe)
		if self.hl == 3:
			eh1 = self.encode_haze1(eds1)
		else:
			eh1 = []
		eds2 = self.encode_ds2(eds1)
		if self.hl == 2 or self.hl == 3:
			eh2 = self.encode_haze2(eds2)
		else:
			eh2 = []
		eds3 = self.encode_ds3(eds2)
		eh3 = self.encode_haze3(eds3)


		return [ecfe, ec1, ec2, ec3], [ehfe, eh1, eh2, eh3]


##########################################################
class HDRUnetDecoderHazy(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, haze_layer = 2, fuse_model='csfm', norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(HDRUnetDecoderHazy, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.fuse_model = fuse_model
		self.hl = haze_layer
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		n_downsampling = 3
		
		if self.fuse_model == 'csfm':
			mult = 2 ** (n_downsampling)
			self.sft_3 = SFT_layer(mult * ngf, mult * ngf)
			if self.hl == 2 or self.hl == 3:
				mult = 2 ** (n_downsampling - 1)
				self.sft_2 = SFT_layer(mult * ngf, mult * ngf)
			if self.hl == 3:
				mult = 2 ** (n_downsampling - 2)
				self.sft_1 = SFT_layer(mult * ngf, mult * ngf)

		elif self.fuse_model == 'cat':
			mult = 2 ** (n_downsampling+1)
			sft_3 = [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
			self.sft_3 = nn.Sequential(*sft_3)

			if self.hl == 2 or self.hl == 3:
				mult = 2 ** (n_downsampling )
				sft_2 = [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
				self.sft_2 = nn.Sequential(*sft_2)

			if self.hl == 3:
				mult = 2 ** (n_downsampling - 1)
				sft_1 = [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
				self.sft_1 = nn.Sequential(*sft_1)

		## feature translation decoder ##
		n_downsampling = 3
		mult = 2 ** n_downsampling
		decode_ft = []
		for i in range(n_blocks):
			decode_ft += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		self.decode_ft = nn.Sequential(*decode_ft)

		offset_channel = ngf
		self.offset_ft = OffsetBlock(ngf * mult, offset_channel, False)
		self.dres_ft = RSABlock(ngf * mult, ngf * mult, offset_channel)

		## 3rd decoder ##
		mult = 2 ** (n_downsampling+1)
		decoder_us3 = [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
		decoder_us3 += [nn.ConvTranspose2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]

		self.decoder_us3 = nn.Sequential(*decoder_us3)

		self.offset_3 = OffsetBlock(int(ngf * mult / 4), offset_channel, False)
		self.dres_3 = RSABlock(int(ngf * mult / 4), int(ngf * mult / 4), offset_channel)

		## 2nd decoder ##
		mult = 2 ** (n_downsampling)
		decoder_us2 = [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
		decoder_us2 += [nn.ConvTranspose2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]

		self.decoder_us2 = nn.Sequential(*decoder_us2)

		self.offset_2 = OffsetBlock(int(ngf * mult / 4), offset_channel, False)
		self.dres_2 = RSABlock(int(ngf * mult / 4), int(ngf * mult / 4), offset_channel)

		## 1st decoder ##
		mult = 2 ** (n_downsampling-1)
		decoder_us1 = [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
		decoder_us1 += [nn.ConvTranspose2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=4, stride=2, padding=1, output_padding=0, bias=use_bias), norm_layer(int(ngf * mult / 4)), nn.ReLU(True)]

		self.decoder_us1 = nn.Sequential(*decoder_us1)

		self.offset_1 = OffsetBlock(int(ngf * mult / 4), offset_channel, False)
		self.dres_1 = RSABlock(int(ngf * mult / 4), int(ngf * mult / 4), offset_channel)

		## final decoder ##
		mult = 2 ** (n_downsampling-2)
		decoder_upfn = [nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
		self.decoder_upfn = nn.Sequential(*decoder_upfn)

		self.offset_fn = OffsetBlock(int(ngf * mult / 4), offset_channel, False)
		self.dres_fn = RSABlock(int(ngf * mult / 4), int(ngf * mult / 4), offset_channel)

		decoder_fn = [nn.ReflectionPad2d(3)]
		decoder_fn += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
		decoder_fn += [nn.Tanh()]
		
		self.decoder_fn = nn.Sequential(*decoder_fn)
	
	def forward(self, c, h):

		if self.fuse_model == 'csfm':
			fuse_3 = self.sft_3(c[3], h[3])
			if self.hl == 3:
				fuse_2 = self.sft_2(c[2], h[2])
				fuse_1 = self.sft_1(c[1], h[1])
			elif self.hl == 2:
				fuse_2 = self.sft_2(c[2], h[2])
				fuse_1 = c[1]
			elif self.hl == 1:
				fuse_2 = c[2]
				fuse_1 = c[1]

		elif self.fuse_model == 'cat':
			fuse_3 = self.sft_3(torch.cat((c[3], h[3]), 1))
			if self.hl == 3:
				fuse_2 = self.sft_2(torch.cat((c[2], h[2]), 1))
				fuse_1 = self.sft_1(torch.cat((c[1], h[1]), 1))
			elif self.hl == 2:
				fuse_2 = self.sft_2(torch.cat((c[2], h[2]), 1))
				fuse_1 = c[1]
			elif self.hl == 1:
				fuse_2 = c[2]
				fuse_1 = c[1]

		dft = self.decode_ft(c[3])
		ofs_ft = self.offset_ft(dft, None)
		rsa_ft = self.dres_ft(dft, ofs_ft)

		concat = torch.cat((fuse_3, dft), 1)
		dup3 = self.decoder_us3(concat)
		ofs_3 = self.offset_3(dup3, ofs_ft)
		rsa_3 = self.dres_3(dup3, ofs_3)

		concat = torch.cat((fuse_2, dup3), 1)
		dup2 = self.decoder_us2(concat)
		ofs_2 = self.offset_2(dup2, ofs_3)
		rsa_2 = self.dres_2(dup2, ofs_2)

		concat = torch.cat((fuse_1, dup2), 1)
		dup1 = self.decoder_us1(concat)
		ofs_1 = self.offset_1(dup1, ofs_2)
		rsa_1 = self.dres_1(dup1, ofs_1)

		concat = torch.cat((c[0], dup1), 1)
		out = self.decoder_upfn(concat)
		ofs_fn = self.offset_fn(dup, ofs_1)
		rsa_fn = self.dres_fn(dup, ofs_fn)

		out = self.decoder_fn(rsa_fn)

		return out

########################################################################
########################################################################



########################################################################
########################################################################
# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class HDRResnetEncoderClear(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(HDRResnetEncoderClear, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		
		model = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.ReLU(True)]
		
		n_downsampling = 2
		for i in range(n_downsampling):
			mult = 2 ** i
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.ReLU(True)]
		
		mult = 2 ** n_downsampling
		for i in range(n_blocks):
			model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		self.model = nn.Sequential(*model)

	def forward(self, input):
		return self.model(input)

class HDRResnetDecoderClear(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(HDRResnetDecoderClear, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		n_downsampling = 2
		for i in range(n_downsampling):
			mult = 2 ** (n_downsampling - i)
			model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
			                             kernel_size=4, stride=2,
			                             padding=1, output_padding=0,
			                             bias=use_bias),
			          norm_layer(int(ngf * mult / 2)),
			          nn.ReLU(True)]
		model += [nn.ReflectionPad2d(3)]
		model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
		model += [nn.Tanh()]
		
		self.model = nn.Sequential(*model)
	
	def forward(self, input):
		return self.model(input)

##############################################################

########################################################################
# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class HDRResnetEncoderHazy(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(HDRResnetEncoderHazy, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		
		model = [nn.ReflectionPad2d(3),
		         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
		                   bias=use_bias),
		         norm_layer(ngf),
		         nn.ReLU(True)]
		
		n_downsampling = 2
		for i in range(n_downsampling):
			mult = 2 ** i
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
			                    stride=2, padding=1, bias=use_bias),
			          norm_layer(ngf * mult * 2),
			          nn.ReLU(True)]
		
		mult = 2 ** n_downsampling
		for i in range(n_blocks):
			model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		self.model = nn.Sequential(*model)

	def forward(self, input):
		return self.model(input)

class HDRResnetDecoderHazy(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=3, padding_type='reflect'):
		assert (n_blocks >= 0)
		super(HDRResnetDecoderHazy, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		n_downsampling = 2
		for i in range(n_downsampling):
			mult = 2 ** (n_downsampling - i)
			model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
			                             kernel_size=4, stride=2,
			                             padding=1, output_padding=0,
			                             bias=use_bias),
			          norm_layer(int(ngf * mult / 2)),
			          nn.ReLU(True)]
		model += [nn.ReflectionPad2d(3)]
		model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
		model += [nn.Tanh()]
		
		self.model = nn.Sequential(*model)
	
	def forward(self, input):
		return self.model(input)

##########################################################################




# Define a resnet block
class ResnetBlock(nn.Module):
	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		super(ResnetBlock, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
	
	def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		conv_block = []
		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
		               norm_layer(dim),
		               nn.ReLU(True)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
		               norm_layer(dim)]
		
		return nn.Sequential(*conv_block)
	
	def forward(self, x):
		out = x + self.conv_block(x)
		return out

'''
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
	def __init__(self, input_nc, output_nc, num_downs, ngf=64,
	             norm_layer=nn.BatchNorm2d, use_dropout=False):
		super(UnetGenerator, self).__init__()
		
		# construct unet structure
		unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
		for i in range(num_downs - 5):
			unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
		unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
		
		self.model = unet_block
	
	def forward(self, input):
		return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
	def __init__(self, outer_nc, inner_nc, input_nc=None,
	             submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False):
		super(UnetSkipConnectionBlock, self).__init__()
		self.outermost = outermost
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		if input_nc is None:
			input_nc = outer_nc
		downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
		                     stride=2, padding=1, bias=use_bias)
		downrelu = nn.LeakyReLU(0.2, True)
		downnorm = norm_layer(inner_nc)
		uprelu = nn.ReLU(True)
		upnorm = norm_layer(outer_nc)
		
		if outermost:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
			                            kernel_size=4, stride=2,
			                            padding=1)
			down = [downconv]
			up = [uprelu, upconv, nn.Tanh()]
			model = down + [submodule] + up
		elif innermost:
			upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
			                            kernel_size=4, stride=2,
			                            padding=1, bias=use_bias)
			down = [downrelu, downconv]
			up = [uprelu, upconv, upnorm]
			model = down + up
		else:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
			                            kernel_size=4, stride=2,
			                            padding=1, bias=use_bias)
			down = [downrelu, downconv, downnorm]
			up = [uprelu, upconv, upnorm]
			
			if use_dropout:
				model = down + [submodule] + up + [nn.Dropout(0.5)]
			else:
				model = down + [submodule] + up
		
		self.model = nn.Sequential(*model)
	
	def forward(self, x):
		if self.outermost:
			return self.model(x)
		else:
			return torch.cat([x, self.model(x)], 1)
'''

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
		super(NLayerDiscriminator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		
		kw = 4
		padw = 1
		sequence = [
			nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
			nn.LeakyReLU(0.2, True)
		]
		
		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
				          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]
		
		nf_mult_prev = nf_mult
		nf_mult = min(2 ** n_layers, 8)
		sequence += [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
			          kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]
		
		sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
		
		if use_sigmoid:
			sequence += [nn.Sigmoid()]
		
		self.model = nn.Sequential(*sequence)
	
	def forward(self, input):
		return self.model(input)


class PixelDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
		super(PixelDiscriminator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		
		self.net = [
			nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
			norm_layer(ndf * 2),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
		
		if use_sigmoid:
			self.net.append(nn.Sigmoid())
		
		self.net = nn.Sequential(*self.net)
	
	def forward(self, input):
		return self.net(input)


class Classifier(nn.Module):
	def __init__(self, input_nc, ndf, norm_layer=nn.BatchNorm2d):
		super(Classifier, self).__init__()
		
		kw = 3
		sequence = [
			nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2),
			nn.LeakyReLU(0.2, True)
		]
		
		nf_mult = 1
		nf_mult_prev = 1
		for n in range(3):
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
				          kernel_size=kw, stride=2),
				norm_layer(ndf * nf_mult, affine=True),
				nn.LeakyReLU(0.2, True)
			]
		self.before_linear = nn.Sequential(*sequence)
		
		sequence = [
			nn.Linear(ndf * nf_mult, 1024),
			nn.Linear(1024, 10)
		]
		
		self.after_linear = nn.Sequential(*sequence)
	
	def forward(self, x):
		bs = x.size(0)
		out = self.after_linear(self.before_linear(x).view(bs, -1))
		return out
#       return nn.functional.log_softmax(out, dim=1)
