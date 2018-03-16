from utils import *

class RNN():
    def __init__(self, input_dim, hid_dim, isTrainable=True):
        self.in_dim = input_dim
        self.hid_dim = hid_dim
        # RNN eqtn s_t = tanh(U*x_t + W*s_(t-1)); yt = softmax(V*s_t)
        self.U = torch.randn((self.hid_dim, self.in_dim))
        self.W = torch.randn((self.hid_dim, self.hid_dim))
        self.V = torch.randn((self.in_dim, self.hid_dim))

        self.gradU = torch.zeros((self.hid_dim, self.in_dim))
        self.gradW = torch.zeros((self.hid_dim, self.hid_dim))
        self.gradV = torch.zeros((self.in_dim, self.hid_dim))

        self.h_genesis = torch.zeros((self.hid_dim, 1))
        self.h_prev = self.h_genesis

        self.cell_h_states = []
        self.cell_outputs = []
        self.bptt_truncate = 1
        self.isTrainable = isTrainable
    
    def get_embedding(self, seq):
        T = seq.size()[1] #seq of shape 1*seq_length
        X = torch.zeros((self.in_dim, T))
        for t,s in enumerate(seq):
            X[s.numpy()[0],t] = 1
        return X

    def forward(self, input_seq):
        # input_seq is a sequence of lenth T
        # each element of the sequence have to be in some form of encoding (usual convention)
        # hence I am using one-hot encoding
        # input_seq of shape (emb_dim, seq_length)
        num_tstep = input_seq.size()[1]
        enc_X = input_seq
        # print('===== enc_X ====== \n', enc_X)
        h_states = torch.zeros((self.hid_dim, num_tstep))
        #since it is many to one output will be only for the final cell
        outs = torch.zeros((self.in_dim, 1))
        for t in range(num_tstep):
            x_inp = enc_X[:,t].contiguous().view(-1,1)
            h_states[:,t] = torch.tanh(torch.mm(self.U, x_inp) + torch.mm(self.W, self.h_prev))
            self.h_prev = h_states[:,t].contiguous().view(-1,1)
            if t == num_tstep-1:
                # print('in forward a : ', self.V)
                # print('in forward b : ', self.h_prev)
                # print('in forward c : ', torch.mm(self.V, self.h_prev))
                outs[:,0] = softmax(torch.mm(self.V, self.h_prev))
        self.cell_h_states = h_states
        self.cell_outputs = outs
        return [self.cell_h_states, self.cell_outputs]
    
    def backward(self, input_x, gradOutput):
        T = input_x.size()[1] #input_x of shape embed_size * seq_length
        # perform forward propagation
        o, s = self.cell_outputs, self.cell_h_states
        delta_o = gradOutput # delta_o only has the gradients wrt to output of last ele of sequence, delta_o.contiguous().view(-1,1) is of shape in_dim * 1
        delta_T = torch.mm(self.V.t(), delta_o.contiguous().view(-1,1)) * (1-(s[:,-1].contiguous().view(-1,1)**2)) #of shape hid_dim * 1
        # Since it's many to one the backpropgation will only happen from the last cell ... and will go back to only bptt_th cell from last
        self.gradV += torch.mm(delta_o.contiguous().view(-1,1), s[:,-1].contiguous().view(-1,1).t()) # at time step t, shape is in_dim * hid_dim
        delta_t = delta_T
        for bptt_step in np.arange(max(0, T-self.bptt_truncate), T)[::-1]:
            self.gradU += torch.mm(delta_t, input_x[:,bptt_step].contiguous().view(-1,1).t())
            self.gradW += torch.mm(delta_t, s[:,bptt_step-1].contiguous().view(-1,1).t())
            delta_t = torch.mm(self.W.t(), delta_t)*(1-s[:,bptt_step-1].contiguous().view(-1,1)**2)
            # print('timestep : {}, gradU : {}'.format(bptt_step, self.gradU))
        #to get the gradients with respect to all of the elements in input sequence
        gradInput = torch.zeros((self.in_dim,T))
        delta_t = delta_T
        for t in np.arange(T)[::-1]:
            # print('inside RNN back a : ', delta_t)
            gradInput[:,t] = torch.mm(self.U.t(), delta_t) #of shape in_dim * 1
            delta_t = torch.mm(self.W.t(), delta_t)*(1-s[:,bptt_step-1].contiguous().view(-1,1)**2)
            delta_t[delta_t != delta_t] = 0
        return gradInput

    def clear_grad(self):
        self.gradU = torch.zeros((self.hid_dim, self.in_dim))
        self.gradW = torch.zeros((self.hid_dim, self.hid_dim))
        self.gradV = torch.zeros((self.in_dim, self.hid_dim))
        return
    
    def __str__(self):
        string = 'RNN'
        return string
    
    def print_param(self):
        print("U :")
        print(self.U)
        print("W :")
        print(self.W)
        print("V :")
        print(self.V)
        print("h_prev :")
        print(self.h_prev)

