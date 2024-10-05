import copy
import numpy as np
import scipy.io as scio

import matplotlib.pyplot as plt

def RSVA( data, ws): # 1D

	filtered_data = copy.deepcopy(data)
	temp_data = copy.deepcopy(data)
	
	step = 100
	for w1 in np.linspace(0,0.5,step):
		for w2 in np.linspace(0,w1,step): # 0<w2<w1<0.5 Monotonic constraint
			a = 1 - 2*w1*np.sinc(ws) - 2*w2*np.sinc(2*ws); # Normalization constraint
			convk = np.array([w2,w1,a,w1,w2])
			
			if filtered_data.ndim == 1:
				temp_data = np.convolve(filtered_data, convk, 'same')

			elif filtered_data.ndim == 2:
				for idx in range(len(filtered_data)):
					temp_data[idx,:] = np.convolve(data[idx,:], convk, 'same')
			else:
				break
						
			filtered_data = np.minimum(temp_data, filtered_data)

	return filtered_data

def SVA1D(data_in, ws):
	g = copy.deepcopy(data_in)
	length = len(g)
	a = np.zeros(length)
	seq = np.zeros(length)
	wmax = 0.5 # maximum w

	L = int(1./ws) # step
	for m in range(L,length-L):
		a[m] = -g[m] / ( g[m-L] + g[m+L] )
		if a[m] < 0:
			seq[m] = g[m]
		elif a[m] >= 0 and a[m] < wmax:
			seq[m] = a[m] * g[m-L] + g[m] + a[m] * g[m+L]
		else:
			seq[m] = wmax * g[m-L] + g[m] + wmax * g[m+L]

	return seq

def SVA2D(data_in, wa, wr):
	data = copy.deepcopy(data_in)
	H, W = data.shape # data shape (Na, Nr)
	wmax = 0.5 # maximum w

	# filtering along the range
	L = int(1./wr) # step
	for h in range(H):
		g = data[h, :]
		seq = np.zeros(W) # save the temporary result
		a = np.zeros(W) # criterion

		for m in range(L,W-L):
			a[m] = -g[m] / ( g[m-L] + g[m+L] )
			if a[m] < 0:
				seq[m] = g[m]
			elif a[m] >= 0 and a[m] < wmax:
				seq[m] = a[m] * g[m-L] + g[m] + a[m] * g[m+L]
			else:
				seq[m] = wmax * g[m-L] + g[m] + wmax * g[m+L]
		
		data[h, :] = seq

	# filtering along the azimuth
	L = int(1./wa) # step
	for w in range(W):
		g = data[:, w]
		seq = np.zeros(H) # save the temporary result
		a = np.zeros(H) # criterion

		for m in range(L,H-L):
			a[m] = -g[m] / ( g[m-L] + g[m+L] )
			if a[m] < 0:
				seq[m] = g[m]
			elif a[m] >= 0 and a[m] < wmax:
				seq[m] = a[m] * g[m-L] + g[m] + a[m] * g[m+L]
			else:
				seq[m] = wmax * g[m-L] + g[m] + wmax * g[m+L]

		data[:, w] = seq

	return data

if __name__ == '__main__':
	
	# load data
	data_path = './exp/signal-point.mat'
	original_data = scio.loadmat(data_path)
	original_data = original_data['signal']

	# filtering
	wa = 0.789434605827874
	wr = 0.714283713575785
	filtered_data = SVA2D( np.real(original_data), wa, wr ) + 1j*SVA2D( np.imag(original_data), wa, wr )
	# filtered_data = RSVA( np.real(original_data), wr ) + 1j*RSVA( np.imag(original_data), wr )

	plt.matshow(np.abs(filtered_data), cmap='hot')
	plt.colorbar()
	plt.show()

