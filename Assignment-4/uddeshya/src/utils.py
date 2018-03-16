import numpy as np
import random
import sys, os, shutil
import torch

class DataLoader():
    def __init__(self, label_path, data_path, batch_size=1):
        self.data_obj = open(data_path, 'r')
        self.lbl_obj = open(label_path, 'r')
        self.data_lines = self.data_obj.readlines()
        self.data_lines = [[int(num_str) for num_str in l.split()] for l in self.data_lines]
        self.lbl_lines = self.lbl_obj.readlines()
        self.lbl_lines = [int(num_str) for num_str in self.lbl_lines]
        self.num_sample = len(self.lbl_lines)
        self.batch_size = batch_size
        #8% of data to be used for validation
        val_idx = self.num_sample-int(self.num_sample*0.08)
        self.train_data = np.array([(self.data_lines[i], self.lbl_lines[i]) for i in range(val_idx)])
        self.val_data = np.array([(self.data_lines[i], self.lbl_lines[i]) for i in range(val_idx, self.num_sample)])
    
    def get_train_batch(self):
        n_train = len(self.train_data)
        idxs = [i for i in range(n_train)]
        while True:
            dbatch_idx = random.sample(idxs, self.batch_size)
            dbatch = self.train_data[dbatch_idx]
            yield dbatch
    
    def get_val_data(self):
        return self.val_data

def softmax(x):
    # print('in softmax a : ', x)
    x = torch.clamp(x, max=80)
    exps = torch.clamp(torch.exp(x), max=1e88)
    exps = torch.clamp(torch.exp(x), min=1e-60)
    # for id,t in enumerate(exps):
    #     v = t == float('inf')
    #     if v.numpy():
    #         print('bitchhhhhhh', id, x[id])
    # print('in softmax b : ', exps)
    # print('in softmax c : ', torch.sum(exps))
    # print('in softmax d : ', exps/torch.sum(exps))
    return exps/torch.sum(exps)

def get_one_hot(emb_dim, input_seq):
    T = input_seq.size()[1] #input_seq of shape (1, seq_length)
    X = torch.zeros((emb_dim, T))
    for t,s in enumerate(input_seq):
        X[s.numpy()[0],t] = 1
    return X

# dpath = '../datasets/train/train_data.txt'
# lpath = '../datasets/train/train_labels.txt'
# D = DataLoader(lpath, dpath)
# print('look at max', np.max(np.max(np.array(D.data_lines))))
# dl = D.get_train_batch()
# a = next(dl)
# print('train batch', a)
# print(type(a[0]))
# # print('val data', D.get_val_data())
