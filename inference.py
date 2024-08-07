import torch
import scipy.io as scio
import matplotlib.pyplot as plt 
import numpy as np 
import math
import copy

from svnn import svnn

INPUTLENGTH = 512

def filterSingleDirection(net, data_single_channel_single_direction):
	data = copy.deepcopy(data_single_channel_single_direction).astype(np.float32)
	data_out = copy.deepcopy(data_single_channel_single_direction)
	for i,seq in enumerate(data):
		norm_factor = np.max(np.abs(seq))
		tensor_in = torch.tensor(seq/norm_factor).unsqueeze(0).unsqueeze(0).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		tensor_out = net(tensor_in).squeeze().cpu().detach().numpy()
		data_out[i,:] = tensor_out * norm_factor

	return data_out 

def filterSingleChannel(net, data_single_channel, rot_cnt=4):
	data = copy.deepcopy(data_single_channel)

	data = filterSingleDirection(net, data)
	data = filterSingleDirection(net, data.T).T
	
	return data

def inference(model_info, data_path, save_path):
	# load model
	net = svnn(model_info['filter_length'])
	net.load_state_dict(torch.load(model_info['model_path'], map_location=torch.device('cpu')))
	net.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
	# load data
	original_data = scio.loadmat(data_path)
	# original_data = original_data['original_data']
	original_data = original_data['signal']
	filtered_data = copy.deepcopy(original_data)
	# deal with original data
	Na, Nr = original_data.shape
	round_Na = math.ceil(Na/INPUTLENGTH)
	round_Nr = math.ceil(Nr/INPUTLENGTH)
	if Na<INPUTLENGTH or Nr<INPUTLENGTH:
		raise ValueError('range and azimuth samples must be more than INPUTLENGTH')

	for r_na in range(round_Na):
		if r_na == round_Na - 1:
			for r_nr in range(round_Nr):
				if r_nr == round_Nr - 1:
					data_patch = original_data[Na-INPUTLENGTH:Na, Nr-INPUTLENGTH:Nr]
					filtered_data[Na-INPUTLENGTH:Na, Nr-INPUTLENGTH:Nr] = filterSingleChannel(net, np.real(data_patch)) + 1j*filterSingleChannel(net, np.imag(data_patch)) 
				else:
					data_patch = original_data[Na-INPUTLENGTH:Na, r_nr*INPUTLENGTH:(r_nr+1)*INPUTLENGTH]
					filtered_data[Na-INPUTLENGTH:Na, r_nr*INPUTLENGTH:(r_nr+1)*INPUTLENGTH] = filterSingleChannel(net, np.real(data_patch)) + 1j*filterSingleChannel(net, np.imag(data_patch)) 
		else:
			for r_nr in range(round_Nr):
				if r_nr == round_Nr - 1:
					data_patch = original_data[r_na*INPUTLENGTH:(r_na+1)*INPUTLENGTH, Nr-INPUTLENGTH:Nr]
					filtered_data[r_na*INPUTLENGTH:(r_na+1)*INPUTLENGTH, Nr-INPUTLENGTH:Nr] = filterSingleChannel(net, np.real(data_patch)) + 1j*filterSingleChannel(net, np.imag(data_patch)) 
				else:
					data_patch = original_data[r_na*INPUTLENGTH:(r_na+1)*INPUTLENGTH, r_nr*INPUTLENGTH:(r_nr+1)*INPUTLENGTH]
					filtered_data[r_na*INPUTLENGTH:(r_na+1)*INPUTLENGTH, r_nr*INPUTLENGTH:(r_nr+1)*INPUTLENGTH] = filterSingleChannel(net, np.real(data_patch)) + 1j*filterSingleChannel(net, np.imag(data_patch)) 
	
	scio.savemat(save_path, {'filtered_data': filtered_data})

if __name__ == '__main__':
	
	model_info = {
					'model_path' : './model/CP_epoch25.pth',
					'filter_length' : 15,
				}
	print(model_info)
	data_path = './exp/CascadeTarget.mat'
	save_path = './exp/CascadeTarget_svnn.mat'

	inference(model_info, data_path, save_path)