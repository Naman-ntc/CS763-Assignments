import torch 
from DataLoader import *
from imports import *
import matplotlib.pyplot as plt

class Trainer():
	def __init__(self, model, dloader, gen_d):
		self.model = model
		self.dloader = dloader
		self.gen_d = gen_d
		self.loss_record = []
		self.step_record = []
		self.train_acc_record = []
		self.val_acc_record = []
		print('trainer initialised')

	def train(self, n_epoch=100, n_iter=1000, rec_interval=100, step_interval=5, l_rate=1e-4, lossClass=Criterion()):
		step = 0
		for eph in range(n_epoch):
			val_acc = self.validation_accuracy()
			train_acc = self.training_accuracy()
			self.train_acc_record.append(train_acc)
			self.val_acc_record.append(val_acc)
			print('===========> starting epoch: {}, validation_accuracy:{}, training_accuracy:{}\n'.format(eph, val_acc, train_acc))
			for i in range(n_iter):
				currentData, currentLabels = next(self.gen_d)
				yPred = self.model.forward(currentData)
				lossGrad, loss = lossClass.backward(yPred, currentLabels)
				
				if i%rec_interval == 0:
					print('epoch:{}, iter:{}, loss:{}'.format(eph, i, loss))
				if step%step_interval == 0:
					self.loss_record.append(loss)
					self.step_record.append(step)

				self.model.clearGradParam()
				self.model.backward(currentData, lossGrad)
				for layer in self.model.Layers:
					if layer.isTrainable:
						layer.weight -= l_rate*layer.gradWeight
						layer.bias -= l_rate*layer.gradBias

				step += 1

	def training_accuracy(self):
		acc = 0
		for i in range(10):
			currentData, currentLabels = next(self.gen_d)
			yPred = self.model.forward(currentData)
			N = currentLabels.size()[0]
			acc += (yPred.max(dim=1)[1].type(torch.LongTensor) == currentLabels.type(torch.LongTensor)).sum()/N
		return acc/10

	def validation_accuracy(self):
		acc = 0
		currentData, currentLabels = self.dloader.get_val_data()
		yPred = self.model.forward(currentData)
		N = currentLabels.size()[0]
		acc += (yPred.max(dim=1)[1].type(torch.LongTensor) == currentLabels.type(torch.LongTensor)).sum()/N
		return acc

	def plot_loss(self):
		plt.plot(self.step_record, self.loss_record)
		plt.xlabel('num of steps')
		plt.ylabel('loss value')
		plt.show()

	def plot_accuracy(self):
		plt.plot(self.train_acc_record)
		plt.plot(self.val_acc_record)
		plt.xlabel('epoch')
		plt.ylabel('accuracy')
		plt.show()

