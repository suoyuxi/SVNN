import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from tqdm import tqdm
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import copy
import random

from dataset import BasicDataset
from svnn import svnn

def eval(net, val_loader, device, save=False):
	# net.eval()#enter val mode, shut down dropout and bn

	loss = 0.0
	signal = 0.0
	with tqdm(total=len(val_loader), desc='Validation round', leave=False) as pbar:
		for i, [sample,gt] in enumerate(val_loader):

			with torch.no_grad():
				sample = torch.autograd.Variable(sample,volatile=True)
				sample = sample.to(device=device, dtype=torch.float32)#load to correspond device		
				labels_pre = net(sample)#forword inference
				labels_pre = labels_pre.to(device=torch.device('cpu'), dtype=torch.float32)
				if save:
					pre = torch.squeeze(labels_pre)
					gt = gt.squeeze()
					scio.savemat('/workspace/sidelobeSuppressing/data_mainlobe/pre/{}.mat'.format(i), {'pre': pre.numpy()})	
				
				loss = loss + torch.mean(torch.abs(gt-labels_pre))
				signal = signal + torch.mean(torch.abs(gt))

	return loss / signal

if __name__ == '__main__':
	#load model
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	svnn_path = 'workdir_mainlobe/CP_epoch96.pth'
	net = svnn(filter_length=5, filter_mode=0, pre_weight=True, is_optim=True)
	net.load_state_dict(torch.load(svnn_path, map_location=device))

	net.to(device=device)
	
	#load dataset
	valset = BasicDataset(is_train=False)
	val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
	print( eval(net, val_loader, device, save=True) )
	