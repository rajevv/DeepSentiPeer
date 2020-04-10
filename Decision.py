#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import predict
from sentence_encoder import *
import predict1


data_padded, label_scale, aspects = predict.prepare_data('../../data/iclr_2017')


# #### (papers, paper obj, review, no.reviews, reviews, decision), aspects_score = data_padded_


x_train, y_train, x_dev, y_dev,x_test, y_test = data_padded


x_paper = x_train[0]
x_review = x_train[4]
# x_num_reviews = x_train[3]
x_decision = x_train[5]


d_paper = x_dev[0]
d_review = x_dev[4]
# d_num_reviews = x_dev[3]
d_decision = x_dev[5]


t_paper = x_test[0]
t_review = x_test[4]
# t_num_reviews = x_test[3]
t_decision = x_test[5]



def get_data(data):
    papers, reviews, decision = data 
    reviews_embedded = embed(reviews)
    papers_embedded = embed(papers)
    sentiment_scores = sentiment(reviews)
    #papers_embedded = np.repeat(papers_embedded, num_reviews, axis = 0)
    decision = np.array(decision).astype(int)
    #decision = np.repeat(decision, num_reviews, axis = 0)
    return papers_embedded, reviews_embedded, sentiment_scores, decision



papers_train, reviews_train, sentiment_train, decision_train = get_data((x_paper, x_review, x_decision))



for i in (papers_train, reviews_train, sentiment_train, decision_train):
    print i.shape



papers_valid, reviews_valid, sentiment_valid, decision_valid = get_data((d_paper, d_review, d_decision))



for i in (papers_valid, reviews_valid, sentiment_valid, decision_valid):
    print i.shape



papers_train = np.pad(papers_train, [(0,0),(0, 1494-666), (0,0)], mode = 'constant', constant_values = 0.0)
                      


papers_valid = papers_valid[:,:1494,:]
reviews_valid = np.pad(reviews_valid, [(0,0),(0, 525-318), (0,0)], mode = 'constant', constant_values = 0.0)
sentiment_valid = np.pad(sentiment_valid, [(0,0),(0, 525-318), (0,0)], mode = 'constant', constant_values = 0.0)



for i in (papers_valid, reviews_valid, sentiment_valid, decision_valid):
    print i.shape



papers_test, reviews_test, sentiment_test, decision_test = get_data((t_paper, t_review, t_decision))



for i in (papers_test, reviews_test, sentiment_test, decision_test):
    print i.shape




papers_test = np.pad(papers_test, [(0,0),(0, 1494-419), (0,0)], mode = 'constant', constant_values = 0.0)
reviews_test = np.pad(reviews_test, [(0,0),(0, 525-309), (0,0)], mode = 'constant', constant_values = 0.0)
sentiment_test = np.pad(sentiment_test, [(0,0),(0, 525-309), (0,0)], mode = 'constant', constant_values = 0.0)




# import predict1


# In[2]:


data_padded_, label_scale_, aspects_ = predict1.prepare_data('../../data/iclr_2018')


# #### (papers, paper obj, review, no.reviews, reviews, decision), aspects_score = data_padded_

# In[15]:


decision = []
for paper in data_padded_[0][1]:
    decision.append(paper.__dict__['ACCEPTED'])
decision = np.array(decision).astype(int)    


# In[3]:


train_, ytrain_ = data_padded_
papers_, _,_,_,reviews_,decision_ = train_


# In[4]:


len(papers_)


# In[9]:


paper_vec, review_vec, sentic_vec, dcsn_vec = get_data((papers_[:500], reviews_[:500], decision_[:500]))


# In[10]:


for i in (paper_vec, review_vec, sentic_vec, dcsn_vec):
    print i.shape


# In[11]:


paper_vec1, review_vec1, sentic_vec1, dcsn_vec1 = get_data((papers_[500:], reviews_[500:], decision_[500:]))


# In[12]:


for i in (paper_vec1, review_vec1, sentic_vec1, dcsn_vec1):
    print i.shape


# In[13]:


max_paper_sent = max(paper_vec1.shape[1], paper_vec.shape[1])
max_review_sent = max(review_vec1.shape[1], review_vec.shape[1])


# In[14]:


paper_vec = np.pad(paper_vec, [(0,0),(0, max_paper_sent-paper_vec.shape[1]), (0,0)], mode = 'constant', constant_values = 0.0)
review_vec1 = np.pad(review_vec1, [(0,0),(0, max_review_sent-review_vec1.shape[1]), (0,0)], mode = 'constant', constant_values = 0.0)
sentic_vec1 = np.pad(sentic_vec1, [(0,0),(0, max_review_sent-sentic_vec1.shape[1]), (0,0)], mode = 'constant', constant_values = 0.0)


# In[15]:


paper_v = np.concatenate((paper_vec, paper_vec1), axis=0)
review_v = np.concatenate((review_vec, review_vec1), axis=0)
sentic_v = np.concatenate((sentic_vec, sentic_vec1), axis=0)
dcsn_v = np.concatenate((dcsn_vec, dcsn_vec1), axis = 0)


# In[16]:


sentic_v = np.pad(sentic_v, [(0,0),(0, 525-sentic_v.shape[1]), (0,0)], mode = 'constant', constant_values = 0.0)
review_v = np.pad(review_v, [(0,0),(0, 525-review_v.shape[1]), (0,0)], mode = 'constant', constant_values = 0.0)


# In[17]:


for i in (paper_v, review_v, sentic_v, dcsn_v):
    print i.shape


# ###  Concatenate 2017 and 2018 dataset

# In[1]:


import numpy as np


# In[2]:


# papers_train = np.load('./serial/iclr2017/train/papers.npy')
# reviews_train = np.load('./serial/iclr2017/train/reviews.npy')
# sentiment_train = np.load('./serial/iclr2017/train/sentic.npy')
# decision_train = np.load('./serial/iclr2017/train/dcsn.npy')

# papers_valid = np.load('./serial/iclr2017/dev/papers.npy')
# reviews_valid = np.load('./serial/iclr2017/dev/reviews.npy')
# sentiment_valid = np.load('./serial/iclr2017/dev/sentic.npy')
# decision_valid = np.load('./serial/iclr2017/dev/dcsn.npy')


papers_test = np.load('./serial/iclr2017/test/papers.npy')
reviews_test = np.load('./serial/iclr2017/test/reviews.npy')
sentiment_test = np.load('./serial/iclr2017/test/sentic.npy')
decision_test = np.load('./serial/iclr2017/test/dcsn.npy')


# paper_v = np.load('./serial/iclr2018/papers.npy')
# review_v = np.load('./serial/iclr2018/reviews.npy')
# sentic_v = np.load('./serial/iclr2018/sentic.npy')
# dcsn_v = np.load('./serial/iclr2018/dcsn.npy')


# In[3]:


for i in (papers_test,reviews_test,sentiment_test,decision_test):
    print i.shape


# In[3]:


train_data = []
for i in zip((papers_train, reviews_train, sentiment_train, decision_train), (paper_v, review_v, sentic_v, dcsn_v)):
    train_data.append(np.concatenate((i[0], i[1]), axis = 0))


# In[4]:


for i in train_data:
    print i.shape


# ### Shuffle the dataset

# In[5]:


from sklearn.utils import shuffle
train_p, train_r, train_s, train_d = shuffle(train_data[0], train_data[1], train_data[2], train_data[3])


# In[6]:


for i in (train_p, train_r, train_s, train_d):
    print i.shape


# ### ACL_2017 Cross-Domain

# In[3]:


import predict1
data_padded_, label_scale_, aspects_ = predict1.prepare_data('../../data/iclr_2017')


# In[4]:


for i in data_padded_[0]:
    print len(i)


# In[5]:


x, y = data_padded_


# In[6]:


for paper in data_padded_[0][1]:
    print paper.__dict__['ACCEPTED']


# In[7]:


none = np.where(np.array(x[5]) == None)


# In[8]:


dcsn = np.array(x[5])
dcsn[none] = True


# In[1]:


# dcsn = dcsn.tolist()
# dcsn


# In[22]:


pt,rt,st,dt = get_data((x[0],x[4],dcsn))


# In[25]:


for i in (pt,rt,st,dt):
    print i.shape


# In[24]:


pt = np.pad(pt, [(0,0),(0, 1494-1373), (0,0)], mode = 'constant', constant_values = 0.0)
rt = np.pad(rt, [(0,0),(0, 525-156), (0,0)], mode = 'constant', constant_values = 0.0)
st = np.pad(st, [(0,0),(0, 525-156), (0,0)], mode = 'constant', constant_values = 0.0)
tt = (pt,rt,st)


# ### Model Definition

# In[26]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
torch.cuda.set_device(7)
device = 'cuda'


# In[27]:


from __future__ import division


# In[28]:


class classify(nn.Module):
    def __init__(self, rh1, ch1):
        super(classify, self).__init__()
        
        num_classes = 2
        
#         self.p3 = nn.Sequential(
#                             nn.Conv1d(in_channels = 512, out_channels = 256, kernel_size = 5),
#                             nn.ReLU()
#                             )
        self.r3 = nn.Sequential(
                            nn.Conv1d(in_channels = 512, out_channels = 64, kernel_size = 5),
                            nn.ReLU()
                            )
    
        self.s1 = nn.Linear(4*525,rh1)

        self.l1 = nn.Linear(64, ch1)
        
        self.l3 = nn.Linear(2*100,100)
        self.l4 = nn.Linear(100, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.7)
        
    def forward(self, paper, review, sentiment):  
        batch_size = paper.shape[0]
#         out_p3 = self.p3(paper)
#         out_p3 = F.max_pool1d(out_p3, out_p3.shape[2])

        out_r3 = self.r3(review)
        out_r3 = F.max_pool1d(out_r3, out_r3.shape[2])    #out_p/r shape = (batch_size, #filters, 1)
        
#         out = torch.cat((out_p3, out_r3), dim = 1)         #out shape = (batch_size, num_filters*kernels, 1)
        out = out_r3
        
        r = self.s1(sentiment.view(batch_size, -1))
        r = self.dropout(r)
        
        out = self.l1(out.view(batch_size, -1))
        out = self.dropout(out)

        out = self.l3(torch.cat((out, r), dim = 1))
        out = self.dropout(out)
        out = self.relu(out)
        out = self.l4(out)
    
        
        return out,r


# In[29]:


# model = classify(100,100).to(device)


# In[30]:


# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# pytorch_total_params


# In[31]:


# optimizer = torch.optim.SGD(model.parameters()) #,weight_decay = 0.0, momentum = 0.9, lr = 0.009)
#loss = torch.nn.CrossEntropyLoss()


# In[29]:


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
        return (self.x1_data[index,:,:], self.x2_data[index,:,:], self.x3_data[index,:,:]), self.y_data[index]
    def __len__(self):
        return self.len


# In[30]:


def load_data(Dset, batch_size, num_workers):
    loader = DataLoader(Dset, batch_size = batch_size, shuffle = False, num_workers = num_workers)
    return loader


# In[34]:


# def accuracy(preds, true):
#     preds = preds.detach().cpu().numpy()
#     true = true.detach().cpu().numpy()
#     labels = np.argmax(preds, axis = 1)
#     return np.sum(np.array(labels == true).astype(int))/float(true.shape[0]), zip(labels, true)


# ###  Train the model

# In[35]:


import time
import json


# In[36]:


trainD = dset((train_p, train_r, train_s), train_d)
trainloader = load_data(trainD, batch_size = 32, num_workers = 1)


# In[37]:


validD = dset((papers_valid, reviews_valid, sentiment_valid), decision_valid)
validloader = load_data(validD, batch_size = 32, num_workers = 1)


# In[38]:


from tensorboardX import SummaryWriter


# In[39]:


def train(model, optimizer, params):
    model.train()
    steps = 0
    loss_log = []
    acc_log = []
    best_val_acc = 0.0
    log_after_interval = params['log_after_interval']
    eval_after_interval = params['eval_after_interval']
    epochs = params['epochs']
    writer = SummaryWriter(comment = 'Decision(r+s),' + str(params['lr']) + ' ' + str(params['l2']))
    for epoch in range(epochs):
        training_loss = []
        training_acc = []
        for i, data in enumerate(trainloader,0):
            (papers, reviews, sentiment), decision = data
            papers = papers.transpose(1,2).float().to(device)
            reviews = reviews.transpose(1,2).float().to(device)
            sentiment = sentiment.transpose(1,2).float().to(device)
            decision = decision.to(device)
            
            optimizer.zero_grad()
            out = model(papers, reviews, sentiment)
            
            pred = (torch.max(out, 1)[1].view(decision.size()).data == decision.data).sum()
            acc = (pred.item()/decision.size()[0])
            
            los = F.cross_entropy(out, decision)
            training_loss.append(los.item())
            training_acc.append(acc)
            loss_log.append(los.item())
            acc_log.append(acc)
            
            los.backward()
            optimizer.step()

            if steps%log_after_interval == 0:
#                 pred = (torch.max(out, 1)[1].view(decision.size()).data == decision.data).sum()
#                 acc = (pred.item()/decision.size()[0])
                print('Epoch[{}/{}] Iteration[{}]-loss: {:.6f} acc: {:.4f}'.format(epoch, epochs, steps, np.average(loss_log), np.average(acc_log)))
                loss_log = []
                acc_log = []
            if steps%eval_after_interval == 0:
                dl, da = evaluate(validloader, model)
                if best_val_acc < da:
                    best_val_acc = da
                    print('Saving model with validation accuracy: {}'.format(da))
                    checkpoint = {'model_state_dict': model.state_dict(),
                                'optimizer_state_dict':optimizer.state_dict(),
                                }
                    with open('./trained_task2/rs/' + params['iteration'] + '.model', 'wb') as f:
                        torch.save(checkpoint, f)
                    with open('./trained_task2/rs/' + params['iteration'] + '.json', 'w') as f:
                        json.dump(params, f)
            steps+=1
        writer.add_scalar('training_loss', np.average(training_loss), epoch)
        writer.add_scalar('training_acc', np.average(training_acc), epoch)
            
            
            


# In[31]:


def evaluate(validloader, model):
    val_loss = []
    val_acc = []
    model.eval()
    for i, data in enumerate(validloader,0):
        (papers, reviews, sentiment), decision = data
        papers = papers.transpose(1,2).float().to(device)
        reviews = reviews.transpose(1,2).float().to(device)
        sentiment = sentiment.transpose(1,2).float().to(device)
        decision = decision.to(device)
        
        out,r = model(papers, reviews, sentiment)
        los = F.cross_entropy(out, decision)
        val_loss.append(los.item())
        pred = (torch.max(out, 1)[1].view(decision.size()).data == decision.data).sum()
        acc = (pred.item()/decision.size()[0])
        val_acc.append(acc)
    print('Evaluation- loss: {:.6f} acc: {:.4f}'.format(np.average(val_loss), np.average(val_acc)))#, pred,decision.size()[0]))
    return np.average(val_loss), np.average(val_acc), out,decision,r


# In[41]:



def Gridtest():

    lrate = [0.5,0.1, 0.07, 0.05, 0.03, 0.01, 0.007, 0.005, 0.003,0.001, 0.0007, 0.0005, 0.0003, 0.0001, 0.00005]
    decay = [0.0001, 0.005, 0.001, 0.05, 0.01, 0.5, 0.1, 0.0, 1.0, 2.0 ]

    for l in lrate:
        for d in decay:
            print('testing with lr {} and l2 {}'.format(l, d))
            params = {
                         'optimizer': 'SGD',
                         'Type': 'Review+Sentiment',
                         'Filter_size': '64 on review',
                         'Dropout': 0.7,
                        'lr':l,
                        'l2': d,
                        'batch-size': 32,
                         'iteration': str(int(time.time())),
                        'epochs': 50,
                        'log_after_interval': 20,
                        'eval_after_interval': 20
            }
            model = classify(100,100).to(device)
            optimizer = torch.optim.SGD(model.parameters(),weight_decay = d, momentum = 0.9, lr = l)
            train(model, optimizer, params)


# In[2]:


# Gridtest()


# ### test the model

# In[32]:


import glob


# In[33]:


for f in glob.glob('./trained_task2/rs/?*.model'):
    checkpoint = torch.load(f)
    testD = dset(tt, dt)
    testloader = load_data(testD, batch_size = 32, num_workers = 1)
    model = classify(100,100).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.load_state_dict(torch.load('prediction.model'))
    _, _ ,p,d,r = evaluate(testloader, model)
###Evaluation- loss: 0.825200 acc: 0.7105


# In[20]:


pred = (torch.max(p, 1)   )[1] #.view(decision.size()).data == decision.data)
# acc = (pred.item()/decision.size()[0])


# In[9]:


# pred


# In[10]:


# d


# In[11]:


# act = r.cpu().detach().numpy()
# act.shape


# In[8]:


# from sklearn.manifold import TSNE
# act_embedded = TSNE(n_components=2).fit_transform(act)
# act_embedded.shape


# In[7]:


# % matplotlib inline
# import matplotlib.pyplot as plt
# cm = plt.cm.get_cmap('RdYlBu')
# # co = pred.cpu().detach().numpy()
# # plt.scatter(act_embedded[:,0], act_embedded[:,1], c=co, marker = markers)
# # ax.set_xlim([-200,200])
# # ax.set_ylim([-200,200])
# # plt.show()
# correct = np.where(pred.cpu().detach().numpy() == d.cpu().detach().numpy())
# wrong = np.where(pred.cpu().detach().numpy() != d.cpu().detach().numpy())
# act_correct = act_embedded[correct]
# act_wrong = act_embedded[wrong]
# f, ax = plt.subplots(figsize = (3.5,2.5))
# plt.scatter(act_correct[:,0], act_correct[:,1], marker='o', c=pred.cpu().detach().numpy()[correct])
# plt.scatter(act_wrong[:,0], act_wrong[:,1], marker='x', c=pred.cpu().detach().numpy()[wrong])


# In[3]:


# co = pred.cpu().detach().numpy()[correct]
# co


# In[ ]:





# In[4]:


# with open('iclr2017_test.txt', 'w') as f:
#     for i in range(len(x_test[1])):
#         f.write(x_test[1][i].__dict__['TITLE']+'\t'+(str(pred[i].item())+','+str(d[i].item())))
#         f.write('\n')


# In[5]:


# checkpoint = {'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict':optimizer.state_dict(),
#                 'batch_size': 32}
# params = {
#              'optimizer': 'Adam',
#              'Type': 'Review+Sentiment',
#              'Filter_size': '64 on review',
#              'Dropout': 0.7,
#              'iteration': str(int(time.time()))
# }


# In[6]:


# with open('./trained_task2/rs/' + params['iteration'] + '.model', 'wb') as f:
#                         torch.save(checkpoint, f)
# with open('./trained_task2/rs/' + params['iteration'] + '.json', 'w') as f:
#         json.dump(params, f)


# In[ ]:




