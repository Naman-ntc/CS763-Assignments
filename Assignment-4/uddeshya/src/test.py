from RNN import *
from Linear import *
from Criterion import *

#loading the Data
dpath = '../datasets/train/train_data.txt'
lpath = '../datasets/train/train_labels.txt'
D = DataLoader(lpath, dpath)
print('look at max', np.max(np.max(np.array(D.data_lines))))
dl = D.get_train_batch() #batch size is by defalt set to one
t_d = next(dl)
t_data_enc = torch.zeros(1, len(t_d[0][0]))
t_data_enc[0,:] = torch.from_numpy(np.array(t_d[0][0]))
t_data_enc = get_one_hot(400, t_data_enc)
t_label_enc = torch.zeros(1, t_d.shape[0])
t_label_enc[0, :] = torch.from_numpy(np.array([t_d[0][1]]))
t_label_enc = get_one_hot(2, t_label_enc)
print(t_data_enc[:,0], t_label_enc[:,0])

l  = RNN(400, 1000)
l.print_param()

# processing one sequence
t_h, t_o = l.forward(t_data_enc)
print('hidden state : ', t_h)
print('outputs : ', t_o)

a_grad = torch.rand(400, 1) #assume grads corresponding to output of first seq
# t = l.get_embedding(a1[0,:].view(1,-1))
t = l.backward(t_data_enc, a_grad)
print('gradInput : ', t.size(), t)
print('gradU : ', l.gradU)
print('gradV : ', l.gradV)
print('gradW : ', l.gradW)

l2 = Linear(400, 2)
t_o2 = l2.forward(t_o)
print('after Dense : ', t_o2)
a_grad2 = CE_criterion().backward(t_o2, t_label_enc)
print('a_grad2 : ', a_grad2[1])
t = l2.backward(t_o, a_grad2[1])
print('gradInput : ', t.size(), t)

t = l.backward(t_data_enc, a_grad)
print('gradInput : ', t.size(), t)
print('gradU : ', l.gradU)
print('gradV : ', l.gradV)
print('gradW : ', l.gradW)

# print('clearing grads ... ')
# l.clear_grad()
# print('gradInput : ', t)
# print('gradU : ', l.gradU)
# print('gradV : ', l.gradV)
# print('gradW : ', l.gradW)
