import json
from predict import *
from net import *
from run_model import *
from sentence_encoder import *
import sys

# return the data and aspect scores for corresponding aspect
def get_data(train, trainlabel, aspect):
    papers, _, reviews, num_reviews,_,_ = train 
    reviews_embedded = embed(reviews)
    papers_embedded = embed(papers)
    papers_embedded = np.repeat(papers_embedded, num_reviews, axis = 0)
    sentiment_scores = sentiment(reviews)
    assert reviews_embedded.shape[0] == len(trainlabel)
    assert papers_embedded.shape[0] == len(trainlabel)
    assert sentiment_scores.shape[0] == len(trainlabel)
    x, y = choose_label((papers_embedded, reviews_embedded, sentiment_scores), trainlabel, label = aspect)
    return x,y

if __name__ == "__main__": 
	args = sys.argv
	
	label = int(args[1])


	print 'Loading Data'
	data_padded, label_scale, aspects = prepare_data('../../data/iclr_2017')

	print aspects[label]

	x_train, y_train, x_dev, y_dev,x_test, y_test = data_padded
	train_x, train_y = get_data(x_train, y_train, label)


	print('papers train: {} \t reviews train: {} \t sentiment train: {}'.format(train_x[0].shape, train_x[1].shape, train_x[2].shape))

	dev_x, dev_y = get_data(x_dev, y_dev, label)
	pd, rd, sd = dev_x
	pd = pd[:,:666,:]
	rd = np.pad(rd, [(0,0),(0, 98-52), (0,0)], mode = 'constant', constant_values = 0.0)
	sd = np.pad(sd, [(0,0),(0, 98-52), (0,0)], mode = 'constant', constant_values = 0.0)
	dev_x = (pd, rd, sd)

	print('papers valid: {} \t reviews valid: {} \t sentiment valid: {}'.format(dev_x[0].shape, dev_x[1].shape, dev_x[2].shape))

	test_x, test_y = get_data(x_test, y_test, label)
	pt = np.pad(test_x[0], [(0,0),(0, 666-419), (0,0)], mode = 'constant', constant_values = 0.0)
	rt = np.pad(test_x[1], [(0,0),(0, 98-50), (0,0)], mode = 'constant', constant_values = 0.0)
	st = np.pad(test_x[2], [(0,0),(0, 98-50), (0,0)], mode = 'constant', constant_values = 0.0)
	test_x = (pt,rt,st)
	
	print('papers test: {} \t reviews test: {} \t sentiment test: {}'.format(test_x[0].shape, test_x[1].shape, test_x[2].shape))

	print 'training starts'
	learning_rate = [0.01, 0.007, 0.005, 0.001]#, 0.007]#[0.9, 0.07, 0.05, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.005, 0.002, 0.001, 0.0009, 0.0005, 0.0001, 0.00007, 0.00005]
	for l in learning_rate:
	    print '#####'
	    print 'training with lr{}'.format(l)
	    print '#####'
	    params = {
	    'iteration' : str(int(time.time())),
	    'Dropout' : 0.5,
	    'l2' : 0.0,
	    'lr' : l,
	    'aspect' : label, 
	    'batch_size' : 32,
	    'comments': 'basic model with one filter review and sentiment, feature level sentiment fusion, merged at last layer\
		            trained with SGD, learning rates varying'
	    }
	    with open('./trained/19012019_/rs/' + params['iteration'] + '.json', 'w') as f:
		json.dump(params, f)
	    
	    model = CNN2(100, 100).to(device)
	    optimizer = torch.optim.SGD(model.parameters(), momentum = 0.9, lr = params['lr'], weight_decay = params['l2'])
	    loss = torch.nn.MSELoss()
	    expr((train_x, train_y, dev_x, dev_y, test_x, test_y),model, params, optimizer, loss)
