import logging

def create_model(opt):
	model = None
	if opt.model == 'cyc': 
		# assert(opt.dataset_mode == 'unaligned')
		from .dh_cycle_gan_model import DH_CYCGAN
		model = DH_CYCGAN()
	elif opt.model == 'cyc_dr':
		from .dh_dr_cycle_gan_model import DH_DR_CYCGAN
		model = DH_DR_CYCGAN()
	elif opt.model == 'cyc_hdr_sspv':
		from .dh_hdr_cycle_sspv_model import DH_HDR_CYC_SSPV
		model = DH_HDR_CYC_SSPV()
	elif opt.model == 'cyc_hdr_uspv':
		from .dh_hdr_cycle_uspv_model import DH_HDR_CYC_USPV
		model = DH_HDR_CYC_USPV()
	elif opt.model == 'test':
		from .test_model import TestModel
		model = TestModel()
	else:
		raise NotImplementedError('model [%s] not implemented.' % opt.model)
	model.initialize(opt)
	logging.info("model [%s] was created" % (model.name()))
	return model
