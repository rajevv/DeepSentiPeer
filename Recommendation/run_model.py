import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from sklearn.metrics import mean_squared_error
import numpy as np
import math

torch.manual_seed(999)
np.random.seed(999)
device = 'cuda'
torch.cuda.manual_seed_all(999)


# Dataset Class
class dset(Dataset):
    def __init__(self, data, y_data):
        x1, x2, x3 = data
        assert x1.shape[0] == x2.shape[0]
        assert x1.shape[0] == x3.shape[0]
        self.len = x1.shape[0]
        self.x1_data = x1
        self.x2_data = x2
        self.x3_data = x3
        self.y_data = y_data
    def __getitem__(self, index):
        return (self.x1_data[index,:,:], self.x2_data[index,:,:], self.x3_data[index,:]), self.y_data[index]
    def __len__(self):
        return self.len

# DataLoader
def load_data(Dset, batch_size, num_workers):
    loader = DataLoader(Dset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    return loader

# Evaluate RMSE
def evaluate(y, y_):
    return math.sqrt(mean_squared_error(y, y_))

def evaluate_(model,loader,loss):
    model.eval()
    loss_log = []
    t = []
    l = []
    for i, data in enumerate(loader,0):
        (papers, reviews, sentiments), y = data
        out = model(papers.transpose(1,2).float().to(device), reviews.transpose(1,2).float().to(device),sentiments.transpose(1,2).float().to(device))
        los = loss(out, y.to(device))
        loss_log.append(los.item())
        for i, (y, y_) in enumerate(zip(y, out)):
                        t.append(y)
                        l.append(y_)
    return np.average(loss_log), evaluate(t,l)
    


def expr(dataset, model, params, optimizer, loss):  #takes an input dictionary of parameters
    #writer = SummaryWriter(comment = 'lr test on basic model with sentic fusion, '+ str(params['aspect']) + ' '+  str(params['lr']))
    train_x, train_y, dev_x, dev_y, test_x, test_y = dataset
    
    trainD = dset(train_x, train_y)
    validD = dset(dev_x, dev_y)
    testD = dset(test_x, test_y)
    dataloader = load_data(trainD, batch_size = params['batch_size'], num_workers = 1)
    validloader = load_data(validD, batch_size = params['batch_size'], num_workers = 1)
    testloader = load_data(testD, batch_size = params['batch_size'], num_workers = 1)
    
    log_after_steps = 10
    eval_after_steps = 10
    epochs = 100
    step = 0
    loss_log = []
    best_val_loss = np.inf
    
    for epoch in range(epochs):
        training_loss = []
	for i, data in enumerate(dataloader,0):
            model.train()
            (papersT, reviewsT, sentimentsT), yT = data
            output = model(papersT.transpose(1,2).float().to(device), reviewsT.transpose(1,2).float().to(device), sentimentsT.transpose(1,2).float().to(device))
            los = loss(output, yT.to(device))
            loss_log.append(los.item())
            training_loss.append(los.item())
            optimizer.zero_grad()
            los.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.5, norm_type=2)
            optimizer.step()

            if step%log_after_steps == 0:
                print('Epoch[{}]/[{}] Iteration[{}] Train Loss: {}'.format(epoch,epochs,step,np.average(loss_log)))
                #writer.add_scalar('train_loss', np.average(loss_log), step)
                loss_log = []

            #if step%eval_after_steps == 0:
	    step+=1

	valid_loss, valid_acc = evaluate_(model,validloader,loss)
		

	print('Epoch[{}]/[{}] Iteration[{}] Valid Loss: {} acc: {}'.format(epoch,epochs,step,valid_loss,valid_acc))
                #writer.add_scalar('valid_loss', np.average(valid_loss_log), step)
		
	if valid_loss < best_val_loss:
		best_val_loss = valid_loss
                print('Saving the best model at epoch: {}, valid_loss: {}'.format(epoch, best_val_loss))
                checkpoint = {'model_state_dict': model.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict(),
                            'val_loss':best_val_loss,
                            'batch_size': params['batch_size']
                             }
                with open('./trained/19012019_/rs/' + params['iteration'] + '.model', 'wb') as f:
                      torch.save(checkpoint, f)
		print('Testing:')
		test_loss, test_acc = evaluate_(model, testloader, loss)
		print('Test Loss: {} acc: {}'.format(test_loss,test_acc))
		    

            #writer.add_scalar('training_loss', np.average(training_loss), epoch)
            





