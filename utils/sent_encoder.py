from sentence_transformers import SentenceTransformer
from sentence_transformers import models
import tensorflow as tf
import tensorflow_hub as hub
import itertools as it
import numpy as np

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

class SentenceEncoder():
	def __init__(self, bert_model):
		self.model = bert_model


	def initialize(self):
		if self.model == 'USE':
			encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/2")



		if self.model == 'scibert_scivocab_uncased':
			word_embedding_model = models.BERT('./../rev_sig/codes/models/scibert_scivocab_uncased/')
			pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
										   pooling_mode_mean_tokens=True,
										   pooling_mode_cls_token=False,
										   pooling_mode_max_tokens=False)
			encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])

		else:
			encoder = SentenceTransformer('bert-base-nli-mean-tokens')
		return encoder

	def Encode(self, x):
		encoder = self.initialize()

		lens = list(map(lambda i:len(i), x))
		x = list(it.chain.from_iterable(x))
		print('Total sentences to be embedded: {}'.format(len(x)))
		emb = encoder.encode(x)
		embedded = []
		zero = [0]*768
		ir = iter(emb)
		for i, l in enumerate(lens):
			z = []
			while len(z) < l:
				z.append(next(ir).tolist())
			embedded.append(z)
		embedded = np.array(list(zip(*list(it.zip_longest(*embedded, fillvalue = zero)))))
		return embedded


	# get the sentiment score for each review
	def sentiment(self, x):  #input list of lists(tokenized sentences for each review)
		print('No. of Reviews: {}'.format(len(x)))
		lens = list(map(lambda i:len(i), x))
		x = list(it.chain.from_iterable(x))   #list of sentences
		print('No. of Sentences: {}'.format(len(x)))
		sentic = []
		zero = [0.0]*4
		ir = iter(x)
		for i in lens:
			score = []
			while len(score) < i:
				score.append(self.Vader(next(ir)))
			sentic.append(score)
		sentic = np.array(list(zip(*list(it.zip_longest(*sentic, fillvalue = zero)))))
		return sentic

	def Vader(self, review):
		polarity = sid.polarity_scores(review)
		sorted_keys = sorted(polarity.keys())
		return [polarity[k] for k in sorted_keys]






if __name__ == "__main__":
	sentences = [
		['Sentences are passed as a list of string.'], ['hi', 'the'], ['you', 'are', 'the', 'best']]

	model = 'scibert_scivocab_uncased'

	sentence_encoder = SentenceEncoder(model)
	embeddings = sentence_encoder.Encode(sentences)
	sentiment = sentence_encoder.sentiment(sentences)
	print(embeddings.shape, sentiment.shape)

	
