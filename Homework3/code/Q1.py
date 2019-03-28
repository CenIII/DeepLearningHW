import torch
import torch.nn as nn
import torch.optim as optim
import os
import bcolz
import numpy as np
import pickle
import os
import random
import tqdm
import copy

def get_glove():
	print('glove start')
	words = []
	idx = 0
	word2idx = {}
	vectors = bcolz.carray(np.zeros(1), rootdir='/Users/CenIII/Downloads/glove.twitter.27B/glove.twitter.27B.100d.dat', mode='w')

	with open('/Users/CenIII/Downloads/glove.twitter.27B/glove.twitter.27B.100d.txt', 'rb') as f:
		for l in f:
			line = l.decode().split()
			if(len(line[1:])==100):
				word = line[0]
				words.append(word)
				word2idx[word] = idx
				idx += 1
				vect = np.array(line[1:]).astype(np.float)
				vectors.append(vect)
			# if idx%100==0:
			# 	print(len(vectors))
	
	print('glove done')
	vectors = bcolz.carray(vectors[1:].reshape((1193513, 100)), rootdir='/Users/CenIII/Downloads/glove.twitter.27B/glove.twitter.27B.100d.dat', mode='w')
	vectors.flush()
	pickle.dump(words, open('/Users/CenIII/Downloads/glove.twitter.27B/glove.twitter.27B.100d_words.pkl', 'wb'))
	pickle.dump(word2idx, open('/Users/CenIII/Downloads/glove.twitter.27B/glove.twitter.27B.100d_idx.pkl', 'wb'))

	vectors = bcolz.open('/Users/CenIII/Downloads/glove.twitter.27B/glove.twitter.27B.100d.dat')[:]
	words = pickle.load(open('/Users/CenIII/Downloads/glove.twitter.27B/glove.twitter.27B.100d_words.pkl', 'rb'))
	word2idx = pickle.load(open('/Users/CenIII/Downloads/glove.twitter.27B/glove.twitter.27B.100d_idx.pkl', 'rb'))

	glove = {w: vectors[word2idx[w]] for w in words}

	return glove

def text_preprocess(pretrained=False):
	if pretrained == True:
		word2vec = np.load('./q1data/glovevec.npy')
		with open('./q1data/gloveDict', "rb") as fp:   #Pickling
			wordDict = pickle.load(fp)
		with open('./q1data/dataset', "rb") as fp:   #Pickling
			dataset = pickle.load(fp)
		return wordDict, word2vec, dataset

	wordDict = {}
	word2vec = []
	dataset = {}
	filelist = os.listdir('./data/')
	wordlist = []
	for file in filelist:
		dataset[file] = {}
		with open('./data/'+file,'r') as f:
			if file!='unlabelled.txt':
				dataset[file]['data'] = []
				dataset[file]['label'] = []
				line = f.readline()
				while line:
					split_line = line.split(' ')
					label = int(split_line[0])
					sentence = split_line[1:]
					dataset[file]['data'].append(sentence)
					dataset[file]['label'].append(label)
					wordlist += sentence
					line = f.readline()
			else:
				dataset[file]['data'] = []
				line = f.readline()
				while line:
					sentence = line.split(' ')
					dataset[file]['data'].append(sentence)
					wordlist += sentence
					line = f.readline()
	vocabs = set(wordlist)
	glove = get_glove()
	cnt=0
	wastewords = []
	for word in vocabs:
		if word in glove:
			word2vec.append(glove[word])
			wordDict[word] = cnt
			cnt += 1
		else:
			wastewords.append(word)
			word2vec.append(np.random.uniform(-1,1,100))
			wordDict[word] = cnt
			cnt += 1
	print(len(wastewords))

	word2vec = np.array(word2vec)
	# with open('./word2vec', "wb") as fp:   #Pickling
	np.save('./data/glovevec.npy',word2vec)
	with open('./data/gloveDict', "wb") as fp:   #Pickling
		pickle.dump(wordDict, fp)

	# convert dataset to indices
	for key in dataset:
		data = dataset[key]['data']
		for sent in data:
			for i in range(len(sent)):
				sent[i] = wordDict[sent[i]]
	with open('./data/dataset', "wb") as fp:   #Pickling
		pickle.dump(dataset, fp)

	return wordDict, word2vec, dataset

def shuffle_data_label(dataset):  # assume data and label exist
	zipped = list(zip(dataset['data'],dataset['label']))
	random.shuffle(zipped)
	dataset['data'],dataset['label'] = zip(*zipped)
	return 

class CNN(nn.Module):
	def __init__(self,vocab_size, hidden_size, kernel_size, num_filters, embedding=None, update_embedding=True):
		super(CNN, self).__init__()
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(vocab_size, hidden_size)
		if embedding is not None:
			self.embedding.weight = nn.Parameter(torch.FloatTensor(embedding))
		self.embedding.weight.requires_grad = update_embedding
		self.kernel_size = kernel_size
		self.conv1d = nn.Conv1d(self.hidden_size,num_filters,self.kernel_size)   # (N, Cin, L)
		self.tail = nn.Sequential(
						nn.ReLU(),
						nn.Linear(num_filters,1),
						nn.Sigmoid()
					) 
		
	def format_input(self, x):
		N = len(x)
		input_lengths = np.zeros(N,dtype=np.int64)
		for i in range(len(x)):
			input_lengths[i] = len(x[i])
		max_len = max(max(input_lengths),self.kernel_size)
		input_var = np.zeros([N,max_len])
		for i in range(len(x)):
			input_var[i,:input_lengths[i]] = np.array(x[i])
		input_var = torch.LongTensor(input_var)
		if torch.cuda.is_available():
			input_var = input_var.cuda()
		return input_var,input_lengths

	def forward(self, x):
		input_var, input_lengths = self.format_input(x)
		# inds = np.argsort(-input_lengths)
		# input_var = input_var[inds]
		# input_lengths = input_lengths[inds]
		# rev_inds = np.argsort(inds)

		embedded = self.embedding(input_var).permute(0,2,1) # change to (N, Cin, L)
		tmp = self.conv1d(embedded)
		tmp = torch.max(tmp,dim=2)[0]
		pred = self.tail(tmp)
		# pred = pred[rev_inds]
		return pred

def predict(model,testdata,save_pred=False,task_id=0):
	preds = np.zeros(len(testdata))
	i = 0
	with torch.no_grad():
		for sent in testdata:
			pred = model([sent])[0]
			# preds[i] = int(pred[0]<pred[1])
			preds[i] = int(pred[0]>0.5)
			i += 1
	if save_pred:
		with open('./Q6exp/predictions_q'+str(task_id)+'.txt','w') as f:
			for pred in preds:
				f.write(str(int(pred))+'\n')
	return preds

def check_accuracy(model, testset):
	data = testset['data']
	label = testset['label']
	preds = predict(model,data)
	acc = np.sum(preds==label)/len(label)
	return acc

# q1 run
def run(model,crit,dataset,task_id,lr=0.01,batchSize=100):
	if torch.cuda.is_available():
		model = model.cuda()
		crit = crit.cuda()
	optimizer = optim.Adam(model.parameters(), lr = lr)
	# shuffle train data

	trainSet = dataset['train.txt']
	valSet = dataset['dev.txt']
	testSet = dataset['test.txt']
	unlabelledSet = dataset['unlabelled.txt']

	trainSize = len(trainSet['data'])
	
	iters = int(trainSize/batchSize)
	
	def adjust_learning_rate(optimizer, epoch, arglr):
		"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
		lr =  arglr * (0.5 ** (epoch // 5))
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
	
	# for epoches
	print('start to train...')
	best_val_acc = 0
	for epoch in range(20):
		shuffle_data_label(trainSet)
		# for iterations
		qdar = tqdm.tqdm(range(iters),total=iters,ascii=True)
		for i in qdar:
			# get batch of data
			inp = list(trainSet['data'][i*batchSize:(i+1)*batchSize])
			label = torch.FloatTensor(trainSet['label'][i*batchSize:(i+1)*batchSize]).unsqueeze(1)
			if torch.cuda.is_available():
				label = label.cuda()
			# feed batch to model
			pred = model(inp)
			# loss
			loss = crit(pred,label)
			# backward
			optimizer.zero_grad()
			loss.backward()
			# step
			optimizer.step()
			qdar.set_postfix(loss= '{:5.4f}'.format(loss))
		
		adjust_learning_rate(optimizer, epoch, lr)
		# validate acc
		val_acc = check_accuracy(model,valSet)
		print('val acc: '+str(val_acc))
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			best_model = copy.deepcopy(model.state_dict())

	model.load_state_dict(best_model)
	torch.save(best_model,'./Q6exp/best_model_'+str(task_id)+'.pt')
	# test acc
	test_acc = check_accuracy(model,testSet)
	print('test acc: '+str(test_acc))
	# predict for unlabeled data and save results.
	predict(model,unlabelledSet['data'],save_pred=True,task_id=task_id)

# main
def main():
	wordDict, word2vec, dataset = text_preprocess(pretrained=True)
	cnn = CNN(len(wordDict), 100, 5, 128, embedding=word2vec, update_embedding=True)
	crit = nn.BCELoss()
	run(cnn,crit,dataset,1,lr=0.001)

if __name__== "__main__":
	main()