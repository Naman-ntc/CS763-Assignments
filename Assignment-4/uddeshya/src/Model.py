import torch
from RNN import *
from Linear import *

class Model(object):
    """docstring for Model"""
    def __init__(self, D, H, isTrain, nLayers=2):
        self.nLayers = nLayers
        self.H = H
        self.D = D
        self.isTrain = isTrain
        #self.embedding = WordEmbedding(V,D)
        self.RNN = RNN(D,H)
        self.fc1 = Linear(D,2)
        self.Layers = [self.fc1,self.RNN]

    def forward(self, input_seq):
        self.h_states, self.out1 = self.RNN.forward(input_seq)
        self.out2 = self.fc1.forward(self.out1.view(-1,1))
        return self.out2

    def backward(self, input_seq, gradOutput):
        gradOut1 = self.fc1.backward(self.out1.view(-1,1), gradOutput.view(-1,1))
        gradInput = self.RNN.backward(input_seq, gradOut1)
        return
    
    def predict(self, input_seq):
        y_hat = self.forward(input_seq)
        pred_label = np.argmax(y_hat)
        return pred_label

    def clearGradParam(self):
        for Layer in self.Layers:
            if Layer.isTrainable:
                Layer.clear_grad()		

    def dispGradParam(self):
        lenn = len(self.Layers)
        for i in range(lenn-1,-1,-1):
            print("Layer : %d"%(i))
            print(self.Layers[i])
            self.Layers[i].print_param()