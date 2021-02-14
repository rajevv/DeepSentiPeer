import os
import json
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys

class Transform(object):
	def __init__(self):
		pass
	def __call__(self, array, max_sents):
		if max_sents < array.shape[0]:
			return torch.from_numpy(array[:max_sents,:])
		else:
			return torch.from_numpy(np.pad(array, [(0, max_sents - array.shape[0]), (0,0)], mode = 'constant', constant_values = 0.0))


class jsonEncoder(object):
	def __init__(self, json_obj=None, mode = None):
		self.json_obj = json_obj
		self.mode = mode

	@classmethod
	def from_json(cls, path, review_filename, mode):
		try:
			return cls(open(os.path.join(path, 'Embeddings', review_filename)), mode=mode)
		except FileNotFoundError:
			return cls(None)

	def __call__(self):
		if not self.json_obj == None:
			encoded = json.load(self.json_obj)
			paper = np.asarray(encoded['paper'])
			reviews = []
			sentiment = []
			if self.mode == 'RECOMMENDATION':
				rec_score = []
				sentiment = []
				for i, review in enumerate(encoded['reviews']):
					reviews.append(np.asarray(review.get('review_text')))
					rec_score.append(review.get('RECOMMENDATION'))
					sentiment.append(review.get('SENTIMENT'))		
				return paper, reviews, rec_score, sentiment
			elif self.mode == 'DECISION':
				pass
			else:
				pass
		else:
			return None
		


class Data(Dataset):
	def __init__(self, data, mode = 'RECOMMENDATION', transform = None, max_paper_sentences=-1, max_review_sentences=-1):
		self.data = data
		self.mode = mode
		self.max_paper_sentences, self.max_review_sentences = max_paper_sentences, max_review_sentences
		self.transform = transform

	@classmethod
	def readData(cls, path, transform = None, jsonEncoder = jsonEncoder(), mode='RECOMMENDATION', slice_=100):
		reviews_dir = os.listdir(os.path.join(path, 'reviews'))[:slice_]
		papers, reviews, rec_scores, reviews_all, decision, sentiment = [], [], [], [], [], []
		max_paper_sents = 0
		max_review_sents = 0
		pbar = tqdm(reviews_dir)
		for review_dir in pbar:
			pbar.set_description("Reading Embeddings...")
			ret = jsonEncoder.from_json(path, review_dir, mode=mode)()
			if ret == None:
				continue
			if ret[0].shape[0] > max_paper_sents:
				max_paper_sents = ret[0].shape[0]

			for i, rev in enumerate(ret[1],0):
				papers.append(ret[0])
				reviews.append(rev)
				sentiment.append(ret[3][i])
				if mode == 'RECOMMENDATION':
					rec_scores.append(int(ret[2][i]))
				elif mode == 'DECISION':
					pass
				else:
					sys.exit("Provide a valid mode from [RECOMMENDATION, DECISION]")
				if rev.shape[0] > max_review_sents:
					max_review_sents = rev.shape[0]	

		if mode == 'RECOMMENDATION':		
			return cls((papers, reviews, sentiment, rec_scores), transform=transform, max_paper_sentences=max_paper_sents, max_review_sentences=max_review_sents, mode=mode)
		if mode == 'DECISION':
			return None
		
	def __getitem__(self, index):
		if self.mode == 'RECOMMENDATION':
			if self.transform:
				return self.transform(np.asarray(self.data[0][index]), self.max_paper_sentences), \
					self.transform(np.asarray(self.data[1][index]), self.max_review_sentences), \
					self.transform(np.asarray(self.data[2][index]), self.max_review_sentences),\
					self.data[3][index]
			else:
				None
		elif self.mode == 'DECISION':
			if self.transform:
				pass
			else:
				pass
		else:
			pass

	def __len__(self):
		return len(self.data)



def getLoaders(data_path = './2018/', mode='RECOMMENDATION', batch_size=8, slice=[-1, -1, -1], valid_path='./2018/'):
	print('Reading the training Dataset...')
	train_dataset = Data.readData(data_path, mode=mode, slice_=slice[0], transform=Transform())
	print('Reading the validation Dataset...')
	valid_dataset = Data.readData(valid_path, mode=mode, slice_=slice[1], transform=Transform())
	
	
	trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=4)
	validloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=True, num_workers=4)
	
	return trainloader, validloader, train_dataset.max_review_sentences, train_dataset.max_paper_sentences
	

	


if __name__ == "__main__":
	trainloader, validloader, _, _ = getLoaders(batch_size=2, slice=[5,5,5])
	print(len(trainloader), len(validloader))
	for i, d in enumerate(zip(*(trainloader, validloader)), 0):
		train, valid = d
		print(train[0].shape)
		print(train[1].shape)
		print(train[2].shape)
		print(train[3])
		break
		
