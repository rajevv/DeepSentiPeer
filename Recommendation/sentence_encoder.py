
"""
__author__    : Rajeev Verma
__date__      : Dec 24th 2018 
__updated__   : Dec 30th 2018
__updated__   : Jan 3rd 2019 sentiment(retuns 3d array)

"""

import numpy as np
import itertools as it
import tensorflow as tf
import tensorflow_hub as hub
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

def run(sentence):
    with tf.Graph().as_default():
        embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        messages = tf.placeholder(dtype=tf.string, shape=[None])
        output = embed(messages)
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            embeddings = session.run(output, feed_dict={messages: sentence})
    return embeddings


# embed the input x and pad appropriately
def embed(x):
	lens = list(map(lambda i:len(i), x))
	x = list(it.chain.from_iterable(x))
	print('Total sentences to be embedded: {}'.format(len(x)))
	emb = run(x)
	embedded = []
	zero = [0]*512
  	ir = iter(emb)
	for i, l in enumerate(lens):
		z = []
    		while len(z)<l:
        		z.append(next(ir).tolist())
    		embedded.append(z)
	embedded = np.array(zip(*list(it.izip_longest(*embedded, fillvalue = zero))))
  	return embedded

    
# get the sentiment score for each review
def sentiment(x):  #input list of lists(tokenized sentences for each review)
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
            score.append(Vader(next(ir)))
        sentic.append(score)
    sentic = np.array(zip(*list(it.izip_longest(*sentic, fillvalue = zero))))
    return sentic

def Vader(review):
    polarity = sid.polarity_scores(review)
    sorted_keys = sorted(polarity.keys())
    return [polarity[k] for k in sorted_keys]

	

		
