from __future__ import division
import numpy as np
import sys
import os
import time
import json
from net import classify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

sys.path.insert(1,os.path.join(sys.path[0],'..'))
from utils.sentence_encoder import *
from utils.predict import *

def get_data(data):
    papers, reviews, decision = data 
    reviews_embedded = embed(reviews)
    papers_embedded = embed(papers)
    sentiment_scores = sentiment(reviews)
    #papers_embedded = np.repeat(papers_embedded, num_reviews, axis = 0)
    decision = np.array(decision).astype(int)
    #decision = np.repeat(decision, num_reviews, axis = 0)
    return papers_embedded, reviews_embedded, sentiment_scores, decision

def padding():
    pass


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
    
    
def load_data(Dset, batch_size, num_workers):
    loader = DataLoader(Dset, batch_size = batch_size, shuffle = False, num_workers = num_workers)
    return loader


def evaluate(validloader, model):
    val_loss = []
    val_acc = []
    validD = dset((papers_valid, reviews_valid, sentiment_valid), decision_valid)
    validloader = load_data(validD, batch_size = 32, num_workers = 1)
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




def train(model, optimizer, params):
    model.train()
    trainD = dset((train_p, train_r, train_s), train_d)
    trainloader = load_data(trainD, batch_size = 64, num_workers = 1)
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
                print('Epoch[{}/{}] Iteration[{}]-loss: {:.6f} acc: {:.4f}'.format(epoch, epochs, steps, np.average(loss_log),                       np.average(acc_log)))
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
            





if __name__ == "__main__":
    
    """
    Since the USE model quering is time consuming, we save the embeddings. If want to query the model again:
                    run $python train.py 1
        While using the saved embeddings:
                    run $python train.py 2
    """
    
    args = sys.argv
    
    
    if args[1] == 1:
        print("Loading the dataset and embeddigs!")
        # Load the dataset (here, ICLR_2017). Repeat it to load other datasets

        data_padded, label_scale, aspects = prepare_data('../../data/iclr_2017')
        x_train, y_train, x_dev, y_dev,x_test, y_test = data_padded

        x_paper, x_review, x_decision = x_train[0], x_train[4], x_train[5]
        # x_num_reviews = x_train[3]

        d_paper, d_review, d_decision = x_dev[0], x_dev[4], x_dev[5]

        t_paper, t_review, t_decision = x_test[0], x_test[4], x_test[5]


        # Get the papers_emeddings, reviews_embeddings, review_sentimentScores and decision
        papers_train, reviews_train, sentiment_train, decision_train = get_data((x_paper, x_review, x_decision))

        papers_valid, reviews_valid, sentiment_valid, decision_valid = get_data((d_paper, d_review, d_decision))

        papers_test, reviews_test, sentiment_test, decision_test = get_data((t_paper, t_review, t_decision))

        """
        The shape of embedded (papers, reviews) is ($a, $b, $c)
            where $a = Total Number of Papers (or Reviews)
                  $b = Maximum number of sentences in a paper across all the Papers (or Reviews)
                  $c = Embedding Dimension (512)
        The $b will be different for different datasets, so while working with other datasets, do the necessary padding
        """
    if args[1] == 2:
        
        print("Using the saved Embeddings!")
        
        papers_train = np.load('../serial/iclr2017/train/papers.npy')
        reviews_train = np.load('../serial/iclr2017/train/reviews.npy')
        sentiment_train = np.load('../serial/iclr2017/train/sentic.npy')
        decision_train = np.load('../serial/iclr2017/train/dcsn.npy')

        papers_valid = np.load('../serial/iclr2017/dev/papers.npy')
        reviews_valid = np.load('../serial/iclr2017/dev/reviews.npy')
        sentiment_valid = np.load('../serial/iclr2017/dev/sentic.npy')
        decision_valid = np.load('../serial/iclr2017/dev/dcsn.npy')


        papers_test = np.load('../serial/iclr2017/test/papers.npy')
        reviews_test = np.load('../serial/iclr2017/test/reviews.npy')
        sentiment_test = np.load('../serial/iclr2017/test/sentic.npy')
        decision_test = np.load('../serial/iclr2017/test/dcsn.npy')
        
    params = {
                        'optimizer': 'SGD',
                        'Type': 'Review+Sentiment',
                        'Filter_size': '64 on review',
                        'Dropout': 0.7,
                        'lr':0.001,
                        'batch-size': 64,
                        'iteration': str(int(time.time())),
                        'epochs': 50,
                        'log_after_interval': 20,
                        'eval_after_interval': 20
    }
    
    
    model = classify(100,100, reviews_train.shape[1]).to(device)
    optimizer = torch.optim.SGD(model.parameters(), momentum = 0.9, lr = 0.001)
    train(model, optimizer, params)
    


