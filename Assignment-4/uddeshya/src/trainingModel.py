from RNN import *
from Linear import *
from Criterion import *
from Model import *

#Dataloader
dpath = '../datasets/train/train_data.txt'
lpath = '../datasets/train/train_labels.txt'
D = DataLoader(lpath, dpath)
# print('look at max', np.max(np.max(np.array(D.data_lines))))
dl = D.get_train_batch() #batch size is by defalt set to one

# val_data = D.get_val_data()
# print("######################################")
# print(val_data.shape)
# print(val_data[0])
# print("######################################")

def eval_train_acc(dloader, model):
    correct_count = 0
    for i in range(10):
        t_d = next(dloader)
        t_data_enc = torch.zeros(1, len(t_d[0][0]))
        t_data_enc[0,:] = torch.from_numpy(np.array(t_d[0][0]))
        t_data_enc = get_one_hot(model.D, t_data_enc) #emb_dimension=400
        t_label_enc = torch.zeros(1, t_d.shape[0])
        t_label_enc[0, :] = torch.from_numpy(np.array([t_d[0][1]]))
        t_label_enc = get_one_hot(2, t_label_enc) #output is in 2 classes

        orig_lbl = np.argmax(t_label_enc)
        y_pred = model.predict(t_data_enc)
        pred_lbl = np.argmax(y_pred)

        if orig_lbl == pred_lbl:
            correct_count += 1
    return correct_count*100/10

def eval_val_acc(D, model):
    correct_count = 0
    val_data = D.get_val_data()
    nS = val_data.shape[0]
    for i in range(nS):
        t_data_enc = torch.zeros(1, len(val_data[i][0]))
        t_data_enc[0,:] = torch.from_numpy(np.array(val_data[i][0]))
        t_data_enc = get_one_hot(model.D, t_data_enc)
        t_label_enc = torch.zeros(1, 1)
        t_label_enc[0, :] = torch.from_numpy(np.array([val_data[i][1]]))
        t_label_enc = get_one_hot(2, t_label_enc)

        orig_lbl = np.argmax(t_label_enc)
        y_pred = model.predict(t_data_enc)
        pred_lbl = np.argmax(y_pred)
        if orig_lbl == pred_lbl:
            correct_count += 1
    return correct_count*100/nS

def train(dloader, model, n_epochs=10, n_iter=100, disp_interval=50, record_interval=5, lr=0.0009):
    for eph in range(n_epochs):
        print('epoch : {} starting ...'.format(eph))
        for itr in range(n_iter):
            t_d = next(dloader)
            t_data_enc = torch.zeros(1, len(t_d[0][0]))
            t_data_enc[0,:] = torch.from_numpy(np.array(t_d[0][0]))
            t_data_enc = get_one_hot(model.D, t_data_enc) #emb_dimension=400
            t_label_enc = torch.zeros(1, t_d.shape[0])
            t_label_enc[0, :] = torch.from_numpy(np.array([t_d[0][1]]))
            t_label_enc = get_one_hot(2, t_label_enc) #output is in 2 classes

            t_out = model.forward(t_data_enc)
            loss, a_grad2 = CE_criterion().backward(t_out, t_label_enc)
            #backprop
            model.backward(t_data_enc, a_grad2)
            for layer in model.Layers:
                if layer.isTrainable:
                    if str(layer) == 'RNN':
                        q1 = layer.U
                        q2 = layer.W
                        q3 = layer.V
                        
                        layer.U -= lr*layer.gradU
                        layer.W -= lr*layer.gradW
                        layer.V -= lr*layer.gradV
                        print('check1 : ', torch.max(q1-layer.U), torch.min(q1-layer.U), torch.max(lr*layer.gradU))
                        print('check2 : ', torch.max(q2-layer.W), torch.min(q2-layer.W), torch.max(lr*layer.gradW))
                        print('check3 : ', torch.max(q3-layer.V), torch.min(q3-layer.V), torch.max(lr*layer.gradV))
                        print('##')
                    elif str(layer) == 'LINEAR':
                        layer.weight -= lr*layer.gradWeight
                        layer.bias -= lr*layer.gradBias

            if itr%disp_interval == 0:
                print('======> epoch : {}, itr : {}, loss : {}, train_acc : {}, val_acc : {}'.format(
                    eph, itr, loss, eval_train_acc(dloader, model), eval_val_acc(D,model)))

M = Model(320, 128, True)
train(dl, M, n_iter=500, disp_interval=10)