import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from sklearn.metrics import mean_squared_error
import numpy as np
import math
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from data_utils import *
from models.recommendation import *
from models.decision import *
import argparse

torch.manual_seed(999)
np.random.seed(999)
device = 'cpu'
#torch.cuda.manual_seed_all(999)


# Evaluate RMSE
def RMSE(y, y_):
	return math.sqrt(mean_squared_error(y, y_))


def evaluate_(mode, model,loader,loss):
	if mode == 'RECOMMENDATION':
		model.eval()
		loss_log = []
		t = []
		l = []
		for i, data in enumerate(loader,0):
			papers, reviews, sentiments, y = data
			out,_ = model(papers.transpose(1,2).float().to(device), reviews.transpose(1,2).float().to(device),sentiments.transpose(1,2).float().to(device))
			los = loss(out.squeeze(), y.float().to(device))
			loss_log.append(los.item())
			for i, (y, y_) in enumerate(zip(y, out)):
							t.append(y)
							l.append(y_)
			return np.average(loss_log), RMSE(t,l)
	elif mode == "DECISION":
		val_loss = []
		val_acc = []
		model.eval()
		for i, data in enumerate(loader,0):
			papers, reviews, sentiment, decision = data
			papers = papers.transpose(1,2).float().to(device)
			reviews = reviews.transpose(1,2).float().to(device)
			sentiment = sentiment.transpose(1,2).float().to(device)
			decision = decision.to(device)
			out,_= model(papers, reviews, sentiment)
			los = F.cross_entropy(out, decision)
			val_loss.append(los.item())
			pred = (torch.max(out, 1)[1].view(decision.size()).data == decision.data).sum()
			acc = (pred.item()/decision.size()[0])
			val_acc.append(acc)
		return np.average(val_loss), np.average(val_acc)


def expr(args, params):
	print(args)

	ckpdir = os.path.join(args.ckpdir, args.mode)
	os.makedirs(ckpdir, exist_ok=True)

	with open(os.path.join(ckpdir, args.exp_name + '.json'), 'w') as f:
		json.dump(params, f)
	
	trainloader, validloader, max_review_sentences, _ = getLoaders(batch_size=args.batch_size, 
																	data_path = args.datadir, 
																	valid_path = args.datadir, 
																	mode = args.mode, 
																	slice=[5,5,5])

	if args.mode == 'RECOMMENDATION':
		model = predict(768, 100, 100, max_review_sentences).to(device)
		optimizer = torch.optim.SGD(model.parameters(), momentum = 0.9, lr = params['lr'], weight_decay = params['l2'])
		loss = torch.nn.MSELoss()
	elif args.mode == 'DECISION':
		model = classify(768, 100, 100, max_review_sentences).to(device)
		optimizer = torch.optim.SGD(model.parameters(), momentum = 0.9, lr = params['lr'], weight_decay = params['l2'])
		loss = nn.CrossEntropyLoss()

	print(model)
	
	log_after_steps = 10
	eval_after_steps = 10
	epochs = 100
	step = 0
	loss_log = []
	acc_log = []
	best_val_loss = np.inf
	best_val_acc = 0.0
	
	for epoch in range(epochs):
		for i, data in enumerate(trainloader,0):
			model.train()
			papersT, reviewsT, sentimentsT, yT = data
			output,_ = model(papersT.transpose(1,2).float().to(device), reviewsT.transpose(1,2).float().to(device), sentimentsT.transpose(1,2).float().to(device))
			if args.mode == "RECOMMENDATION":
				los = loss(output.squeeze(), yT.float().to(device))
				loss_log.append(los.item())
				optimizer.zero_grad()
				los.backward()
				optimizer.step()
				nn.utils.clip_grad_norm_(model.parameters(), 2.5, norm_type=2)
				if step%log_after_steps == 0:
					print('Epoch[{}]/[{}] Iteration[{}] Train Loss: {}'.format(epoch,epochs,step,np.average(loss_log)))
					loss_log = []
			if args.mode == "DECISION":
				los = loss(output.squeeze(), yT.to(device))
				loss_log.append(los.item())
				optimizer.zero_grad()
				los.backward()
				optimizer.step()
				pred = (torch.max(output, 1)[1].view(yT.size()).data == yT.data).sum()
				acc = (pred.item()/yT.size()[0])
				acc_log.append(acc)
				if step%log_after_steps == 0:
					print('Epoch[{}]/[{}] Iteration[{}] Train Loss: {} Training Accuracy: {}'.format(epoch,epochs,step,np.average(loss_log),np.average(acc_log)))
					loss_log = []
					acc_log = []
			step+=1

		valid_loss, valid_acc = evaluate_(args.mode,model,validloader,loss)
		

	print('Epoch[{}]/[{}] Iteration[{}] Valid Loss: {} metric(Acc or RMSE based on mode): {}'.format(epoch,epochs,step,valid_loss,valid_acc))
		
	if valid_loss < best_val_loss:
		best_val_loss = valid_loss
		print('Saving the best model at epoch: {}, valid_loss: {}'.format(epoch, best_val_loss))
		checkpoint = {'model_state_dict': model.state_dict(),
					'optimizer_state_dict':optimizer.state_dict(),
					'val_loss':best_val_loss,
					'batch_size': params['batch_size']
					 }
		with open(os.path.join(ckpdir, args.exp_name + '.model'), 'wb') as f:
			  torch.save(checkpoint, f)

			
def main(args):
		params = {
		'Dropout' : args.dropout,
		'l2' : args.l2,
		'lr' : args.learning_rate, 
		'batch_size' : args.batch_size
		}

		expr(args, params)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--batch_size',
						default=32,
						type=int,
						help='batch size to train the model')
	parser.add_argument('--dropout',
						default=0.5,
						type=float,
						help='dropout probability')
	parser.add_argument('--l2',
						default=0.007,
						type=float,
						help='l2 weight decay penalty')
	parser.add_argument('--learning_rate',
						default=0.001,
						type=float,
						help='learning rate for the gradient based Algorithm')
	parser.add_argument('--mode',
						default='RECOMMENDATION',
						type=str,
						help='Task mode, choose from [RECOMMENDATION, DECISION]')
	parser.add_argument('--datadir',
						default='./2018',
						type=str,
						help='Path to the Dataset')
	parser.add_argument('--ckpdir',
						default='./MODELS',
						type=str,
						help='Path to save the trained models')
	parser.add_argument('--exp_name',
						default='default',
						type=str,
						help='Name of the experiment, model and params will be saved with this name')

	args = parser.parse_args()
	main(args)


