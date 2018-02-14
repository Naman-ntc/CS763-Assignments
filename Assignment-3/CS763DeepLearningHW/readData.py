import torchfile as tf 
import numpy as np 
import torch

npLabels = tf.load('/Users/sahil/Desktop/sem4/computerVision/assgn3/labels.bin')
npData = tf.load('/Users/sahil/Desktop/sem4/computerVision/assgn3/data.bin')

labels = torch.from_numpy(npLabels[:500])
data = torch.from_numpy(npData[:500])

labels = labels.type(torch.FloatTensor)
data = data.type(torch.FloatTensor)