import scipy.io as scio
import os
import random
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

class BasicDataset(Dataset):
	def __init__(self, is_train=True):
		super(BasicDataset, self).__init__()

		if is_train:
			self.sample_path = 'data/train/sample/'
			self.gt_path = 'data/train/gt/'
		else:
			self.sample_path = 'data/val/sample/'
			self.gt_path = 'data/val/gt/'

		self.length = len(os.listdir(self.sample_path))
		self.file_name = [str(x) + '.mat' for x in range(self.length)]

	def __len__(self):
		return self.length

	def __getitem__(self, i):
		file_name = self.file_name[i]
		sample_path = self.sample_path + file_name
		gt_path = self.gt_path + file_name

		sample_mat = scio.loadmat(sample_path)
		sample = sample_mat['sample']
		gt_mat = scio.loadmat(gt_path)
		gt = gt_mat['gt']

		return torch.from_numpy(sample).type(torch.FloatTensor).permute(1,0), torch.from_numpy(gt).type(torch.FloatTensor).permute(1,0)

if __name__ == '__main__':
	Dataset = BasicDataset(is_train=True)
	val_loader = DataLoader(Dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
	print(Dataset[0][0].size())
	print(len(Dataset))

	for item in val_loader:
		print(item[0].size())
		break