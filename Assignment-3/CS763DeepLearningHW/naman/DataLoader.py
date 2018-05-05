import torchfile as torchf 
import numpy as np 
import torch
import random

class DataLoader():
	def __init__(self, label_path, data_path, batch_size):
		self.batch_size = batch_size
		self.labels = torchf.load(label_path)
		self.n_total = len(self.labels)
		self.labels = torch.from_numpy(self.labels).type(torch.FloatTensor)
		self.data = torchf.load(data_path).reshape(self.n_total, -1)
		self.data = torch.from_numpy(self.data).type(torch.FloatTensor)

		self.data_mean = self.data.mean(dim=0)
		self.data_std = self.data.std(dim=0)
		self.data = (self.data-self.data_mean)/self.data_std
		#validation_set
		self.val_data = self.data[:1000, :]
		self.val_labels = self.labels[:1000]
		#training set
		self.data = self.data[1000:, :]
		self.labels = self.labels[1000:]
		self.n_total = len(self.labels)

	def get_batch(self):
		l_idx = [i for i in range(self.n_total)]
		while True:
			dbatch_idx = random.sample(l_idx, self.batch_size)
			dbatch = self.data[dbatch_idx, :]
			dlabel = self.labels[dbatch_idx]
			yield (dbatch, dlabel)

	def get_val_data(self):
		return (self.val_data, self.val_labels)
