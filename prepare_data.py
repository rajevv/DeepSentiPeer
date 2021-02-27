import os
import glob
import json
import torch
import numpy as np
from collections import defaultdict
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from utils.Review import Review
from utils.Paper import Paper
from utils.ScienceParse import ScienceParse
from utils.ScienceParseReader import ScienceParseReader

from utils.sent_encoder import SentenceEncoder

class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)



def writeToFile(matrix, review):
	np.savetxt(review + '.txt', matrix, fmt='%.2f')

def Encode(sentences):
	sentence_encoder = SentenceEncoder('scibert_scivocab_uncased')
	sentence_embeddings = sentence_encoder.Encode(sentences)
	sentiment = sentence_encoder.sentiment(sentences)
	return sentence_embeddings, sentiment


def Embeddings(paper_sent, reviews_sents, test = False, write=False):
	reviews_sents.extend([paper_sent])
	embeddings, sentiment = Encode(reviews_sents)

	return embeddings[-1,:,:], embeddings[:-1,:,:], sentiment[:-1,:,:]


def getEmbeddings(d,paper=None,path=None,filename=None, decision=None):
	paper_emb, reviews_emb, review_sentiment = Embeddings(sent_tokenize(d['paper_content']), [sent_tokenize(r) for r in d['reviews_content']])
	json_obj = {}
	json_obj['reviews'] = []
	json_obj['paper'] = paper_emb
	json_obj['DECISION'] = decision
	for i,review in enumerate(paper.REVIEWS):
		rev_ = {}
		rev_['review_text'] = reviews_emb[i,:,:]
		rev_['RECOMMENDATION'] = review.RECOMMENDATION
		rev_['SENTIMENT'] = review_sentiment[i,:,:]
		json_obj['reviews'].append(rev_)
	 
	with open(os.path.join(path, filename), 'w') as f:
				f.write(json.dumps(json_obj, indent=8, ensure_ascii=False, cls=NumpyEncoder))


	


def preprocess(input, only_char=False, lower=False, stop_remove=False, stemming=False):
	if lower: input = input.lower()
	if only_char:
		tokenizer = RegexpTokenizer(r'\w+')
		tokens = tokenizer.tokenize(input)
		input = ' '.join(tokens)
	sents = sent_tokenize(input)
	tokens = word_tokenize(input)
	if stop_remove:
		tokens = [w for w in tokens if not w in stopwords.words('english')]
	return " ".join(tokens)


def prepare_data(**kwargs):
	print(path)
	data_type = path.split('/')[-1]
	print('Reading datasets...')
	datasets = ['']
	paper_content_all = []
	review_content_all = []

	data = defaultdict(list)

	for dataset in datasets:
		paper_content = []
		review_content = []
		review_dir = os.path.join(path, dataset, 'reviews/')
		scienceparse_dir = os.path.join(path, dataset, 'parsed_pdfs/')
		paper_json_filenames = glob.glob('{}/*.json'.format(review_dir))
		emb_dir = os.path.join(path,dataset,'Embeddings/')
		if not os.path.exists(emb_dir):
			emb_present = False
			os.makedirs(emb_dir)
		else:
			emb_present = True


		if emb_present == False:
			print("Generate Embeddings...")
			papers = []
			for paper_json_filename in paper_json_filenames:
		
				d = {}
				paper = Paper.from_json(paper_json_filename)
				paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(paper.ID,paper.TITLE,paper.ABSTRACT,scienceparse_dir)
				#say some review file does not correspond to the paper file
				if paper.SCIENCEPARSE == None:
					print(paper_json_filename)
					continue
				
				review_contents = []
				reviews = []
				for review in paper.REVIEWS:
					review_contents.append(
						preprocess(review.COMMENTS, only_char=False, lower=True,stop_remove=False))
					reviews.append(review.COMMENTS)

				d['paper_content'] = preprocess(
					paper.SCIENCEPARSE.get_paper_content(), only_char=False, lower=True, stop_remove=False)
				d['reviews_content'] = review_contents
				d['reviews'] = reviews
				data[dataset].append(d)
				try:
					decision= int(paper.ACCEPTED) 
				except:
					decision=0
				getEmbeddings(d, paper=paper, path=emb_dir, filename=paper_json_filename.split('/')[-1], decision=decision)
				with open(os.path.join(emb_dir, paper_json_filename.split('/')[-1]), 'r') as f:
					fl = json.load(f)
					papers.append(fl)

			return papers
		else:
			print("Read Embeddings...")
			papers = []
			for paper_json_filename in paper_json_filenames:
				with open(os.path.join(emb_dir, paper_json_filename.split('/')[-1]), 'r') as f:
					fl = json.load(f)
					papers.append(fl)
			return papers



if __name__ == "__main__":
	path = './2018'
	ret = prepare_data(path = path)
