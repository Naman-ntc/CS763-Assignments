import torch

train_data = open("../Small-Data/train_data.txt","r")
train_labels = open("../Small-Data/train_labels.txt","r")

data = train_data.readlines()
labels = train_labels.readlines()
num_sequences = len(data)


for i in range(num_sequences):
	data[i] = torch.DoubleTensor([int(x) for x in data[i].split()])
	labels[i] = int(labels[i].split()[0])


labels = torch.DoubleTensor(labels)


train_data.close()
train_labels.close()