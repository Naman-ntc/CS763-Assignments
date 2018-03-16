import torch

train_data = open("../Small-Data/train_data.txt","r")
train_labels = open("../Small-Data/train_labels.txt","r")

dictionary_file = open("../Small-Data/for_dict.txt")

dictionary = {}

dictionary_opened = dictionary_file.readlines()

for i in range(153):
	dictionary[int(dictionary_opened[i].split()[0])] = i
	

data = train_data.readlines()
labels = train_labels.readlines()
num_sequences = len(data)


for i in range(num_sequences):
	data[i] = torch.DoubleTensor([dictionary[int(x)] for x in data[i].split()])
	labels[i] = int(labels[i].split()[0])


labels = torch.DoubleTensor(labels)


train_data.close()
train_labels.close()

