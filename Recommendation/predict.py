"""
 predict review scores of each aspect (e.g.,recommendation, clarity, impact, etc)
"""
import sys,os,json, glob,pickle,operator,re,time,logging,shutil,pdb,math
from collections import Counter,OrderedDict,defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import dropwhile
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import cPickle as pkl
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error,accuracy_score

from data_helper import load_embeddings,batch_iter, pad_sentence, progress
from pred_models import CNN,RNN,DAN
from config import CNNConfig, RNNConfig, DANConfig

sys.path.insert(1,os.path.join(sys.path[0],'..'))
from models.Review import Review
from models.Paper import Paper
from models.ScienceParse import ScienceParse
from models.ScienceParseReader import ScienceParseReader

import itertools as it
import pickle
import nltk

def models(model_text):
  if model_text == 'cnn': return CNN,CNNConfig
  elif model_text == 'rnn': return RNN,RNNConfig
  elif model_text == 'dan': return DAN,DANConfig
  else: return None,None


def preprocess(input, only_char=False, lower=False, stop_remove=False, stemming=False):
  #input = re.sub(r'[^\x00-\x7F]+',' ', input)
  if lower: input = input.lower()
  if only_char:
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)
    input = ' '.join(tokens)
  tokens = word_tokenize(input)
  if stop_remove:
    tokens = [w for w in tokens if not w in stopwords.words('english')]

  # also remove one-length word
  tokens = [w for w in tokens if len(w) > 1]
  return " ".join(tokens)


def evaluate(y, y_):
  return math.sqrt(mean_squared_error(y, y_))


def prepare_data(
    data_dir,
    vocab_path='vocab',
    max_vocab_size = 20000,
    max_len_paper=1000,
    max_len_review=200):


  data_type = data_dir.split('/')[-1]
  vocab_path += '.' + data_type
  if max_vocab_size: vocab_path += '.'+str(max_vocab_size)
  vocab_path = data_dir +'/'+ vocab_path

  label_scale = 5
  if 'iclr' in data_dir.lower():
    fill_missing = False
    aspects = ['RECOMMENDATION', 'SUBSTANCE', 'APPROPRIATENESS','MEANINGFUL_COMPARISON','SOUNDNESS_CORRECTNESS','ORIGINALITY','CLARITY', 'IMPACT', 'RECOMMENDATION_ORIGINAL']
    review_dir_postfix = ''
  elif 'acl' in data_dir.lower():
    fill_missing = True
    aspects = ['RECOMMENDATION', 'SUBSTANCE', 'APPROPRIATENESS','MEANINGFUL_COMPARISON','SOUNDNESS_CORRECTNESS','ORIGINALITY','CLARITY','IMPACT', 'REVIEWER_CONFIDENCE' ]
    review_dir_postfix = ''
  else:
    print 'wrong dataset:',data_dir
    sys.exit(1)


  #Loading datasets
  print 'Reading datasets..'
  datasets = ['train','dev','test']
  paper_content_all = []
  review_content_all = []

  data = defaultdict(list)
  for dataset in datasets:

    review_dir = os.path.join(data_dir,  dataset, 'reviews%s/'%(review_dir_postfix))
    scienceparse_dir = os.path.join(data_dir, dataset, 'parsed_pdfs/')
    model_dir = os.path.join(data_dir, dataset, 'model/')
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    paper_json_filenames = sorted(glob.glob('{}/*.json'.format(review_dir)))

    # add all paper/review content to generate corpus for buildinb vocab
    paper_content = []
    review_content = []
    for paper_json_filename in paper_json_filenames:
      d = {}
      paper = Paper.from_json(paper_json_filename)
      paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(paper.ID, paper.TITLE, paper.ABSTRACT, scienceparse_dir)

      review_contents = []
      reviews = []
      for review in paper.REVIEWS:
        review_contents.append(review.COMMENTS)
          #preprocess(review.COMMENTS, only_char=False, lower=True, stop_remove=False))
        reviews.append(review)
      d['paper'] = paper
      d['paper_content'] = paper.SCIENCEPARSE.get_paper_content() #preprocess( #, only_char=False, lower=True,stop_remove=False)#
      d['reviews_content'] = review_contents
      d['reviews'] = reviews
      data[dataset].append(d)

  print 'Total number of papers %d' %(np.sum([len(d) for _,d in data.items()]))
  print 'Total number of reviews %d' %(np.sum([len(r['reviews']) for _,d in data.items() for r in d ]))
    
  # Loading DATA
  print 'Reading reviews from...'
  data_padded = []
  for dataset in datasets:

    ds = data[dataset]
    papers = []
    x_paper = [] #[None] * len(reviews)
    x_review = [] #[None] * len(reviews)
    y = [] #[None] * len(reviews)
    num_reviews = []
    x_reviews = []
    decision = []
    for d in ds:
      paper = d['paper']
      paper_content = d['paper_content']
      reviews_content = d['reviews_content']
      reviews = d['reviews']
      decision.append(paper.ACCEPTED)
      papers.append(paper)
      paper_sent = nltk.sent_tokenize(paper_content)
      reviews_sent = nltk.sent_tokenize(' '.join(reviews_content))
      x_paper.append(paper_sent)
      x_reviews.append(reviews_sent)
      num_reviews.append(len(reviews))
      for rid, (review_content, review) in enumerate(zip(reviews_content,reviews)):
         yone = [np.nan] * len(aspects)
	 review_sent = nltk.sent_tokenize(review.__dict__['COMMENTS'])
         for aid,aspect in enumerate(aspects):
            if aspect in review.__dict__ and review.__dict__[aspect] is not None:
               yone[aid] = float(review.__dict__[aspect])
         x_review.append(review_sent)
         y.append(yone)

    y = np.array(y, dtype=np.float32)
    # add average value of missing aspect value
    if fill_missing:
      col_mean = np.nanmean(y,axis=0)
      inds = np.where(np.isnan(y))
      y[inds] = np.take(col_mean, inds[1])

    data_padded.append((x_paper,papers, x_review, num_reviews, x_reviews, decision))
    data_padded.append(y)

  return data_padded,label_scale,aspects



def choose_label(x,y, size=5, label=False):

  # [size x 9]
  y = np.array(y)

  # (1) only choose label
  if label is not False and label >= 0:
    y = y[:,[label]]

  # (2) remove None/Nan examples ##x[~np.isnan(y).flatten()]	
  x = (
    	x[0][~np.isnan(y).flatten()],
    	x[1][~np.isnan(y).flatten()],
	x[2][~np.isnan(y).flatten()]
  	)

  y = y[~np.isnan(y)]
  y = np.reshape(y, (-1,1))

  assert x[0].shape[0] == y.shape[0]
  assert x[1].shape[0] == y.shape[0]
  assert x[2].shape[0] == y.shape[0]

  return x,y  


#if __name__ == "__main__": #main(sys.argv)

