import sys

import torch.utils.data
from data.base_data_loader import BaseDataLoader

sys.path.append('D:/dzhao/dehazing_360/SSDH-HDR/')



def CreateDataLoader(opt):
	data_loader = CustomDatasetDataLoader()
	print(data_loader.name())
	data_loader.initialize(opt)
	return data_loader


def CreateDataset(opt):
	dataset = None
	if opt.dataset_mode == 'dh_hdr':

		from data.dh_hdr import DH_HDR
		dataset = DH_HDR()

	elif opt.dataset_mode == 'dh_hdr_uspv':

		from data.dh_hdr_uspv import DH_HDR_USPV
		dataset = DH_HDR_USPV()

	elif opt.dataset_mode == 'dh_hdr_sspv':

		from data.dh_hdr_sspv import DH_HDR_SSPV
		dataset = DH_HDR_SSPV()

	elif opt.dataset_mode == 'dh_hdr_sspv_ms':

		from data.dh_hdr_sspv_ms import DH_HDR_SSPV_MS
		dataset = DH_HDR_SSPV_MS()

	else:
		raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)
	
	print("dataset [%s] was created" % (dataset.name()))
	dataset.initialize(opt)
	return dataset


class CustomDatasetDataLoader(BaseDataLoader):
	def name(self):
		return 'CustomDatasetDataLoader'
	
	def initialize(self, opt):
		BaseDataLoader.initialize(self, opt)
		self.dataset = CreateDataset(opt)
		self.dataloader = torch.utils.data.DataLoader(
			self.dataset,
			batch_size=opt.batchSize,
			shuffle=not opt.sb,
			num_workers=int(opt.nThreads))
	
	def load_data(self):
		return self
	
	def __len__(self):
		return min(len(self.dataset), self.opt.max_dataset_size)
	
	def __iter__(self):
		for i, data in enumerate(self.dataloader):
			if i * self.opt.batchSize >= self.opt.max_dataset_size:
				break
			yield data
