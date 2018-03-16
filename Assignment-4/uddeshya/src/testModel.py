from RNN import *
from Linear import *
from Criterion import *
from Model import *

#loading the Data
dpath = '../datasets/train/train_data.txt'
lpath = '../datasets/train/train_labels.txt'
D = DataLoader(lpath, dpath)
# print('look at max', np.max(np.max(np.array(D.data_lines)))) # 311
dl = D.get_train_batch() #batch size is by defalt set to one
t_d = next(dl)
t_data_enc = torch.zeros(1, len(t_d[0][0]))
t_data_enc[0,:] = torch.from_numpy(np.array(t_d[0][0]))
t_data_enc = get_one_hot(400, t_data_enc)
t_label_enc = torch.zeros(1, t_d.shape[0])
t_label_enc[0, :] = torch.from_numpy(np.array([t_d[0][1]]))
t_label_enc = get_one_hot(2, t_label_enc)
print(t_data_enc[:,0], t_label_enc[:,0])

M = Model(400, 1000, True)
M.dispGradParam()

t_o = M.forward(t_data_enc)
print('output of model : ', t_o)
loss, a_grad2 = CE_criterion().backward(t_o, t_label_enc)
print('loss : ', loss)
print('a_grad2 : ', a_grad2)
print('backpropagating through the model ...')
M.backward(t_data_enc, a_grad2)
M.dispGradParam()
print(M.predict(t_data_enc), t_label_enc)
