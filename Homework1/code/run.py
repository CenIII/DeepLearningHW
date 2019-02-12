from logistic import LogisticClassifier
from svm import SVM
from softmax import SoftmaxClassifier
from cnn import ConvNet
from solver import Solver
import pickle
import gzip
import numpy as np
import sys
import os

def runLogistic():
  with open('../data.pkl', "rb" ) as f:
    X,Y = pickle.load(f,encoding='latin1')
  if len(sys.argv)>1:
    exp_name = sys.argv[1]
  else:
    exp_name = 'exp'
  os.makedirs(exp_name,exist_ok=True)
  logger = open("./"+exp_name+"/log","w")
  data = {
      'X_train': X[:500], # training data
      'y_train': Y[:500], # training labels
      'X_val': X[500:750],# validation data
      'y_val': Y[500:750]# validation labels
  }
  model = LogisticClassifier(input_dim=20, hidden_dim=1000, reg=0.004)
  solver = Solver(model, data, logger,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 0.005,
                    },
                    lr_decay=1,
                    num_epochs=200, batch_size=50,
                    print_every=100)
  solver.train()
  test_acc = solver.check_accuracy(X[750:],Y[750:])
  toprint = "test_acc: "+str(test_acc)
  print(toprint)

def runSVM():
  with open('../data.pkl', "rb" ) as f:
    X,Y = pickle.load(f,encoding='latin1')
  if len(sys.argv)>1:
    exp_name = sys.argv[1]
  else:
    exp_name = 'exp'
  os.makedirs(exp_name,exist_ok=True)
  logger = open("./"+exp_name+"/log","w")

  # Y[Y==0] = -1
  data = {
      'X_train': X[:500], # training data
      'y_train': Y[:500], # training labels
      'X_val': X[500:750],# validation data
      'y_val': Y[500:750]# validation labels
  }
  model = SVM(input_dim=20, hidden_dim=1000, reg=0.0)
  solver = Solver(model, data, logger,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 0.1,
                    },
                    lr_decay=1,
                    num_epochs=120, batch_size=50,
                    print_every=100)
  solver.train()
  test_acc = solver.check_accuracy(X[750:],Y[750:])
  toprint = "test_acc: "+str(test_acc)
  print(toprint)

def runSoftmax():
  # Load the dataset
  f = gzip.open('../mnist.pkl.gz', 'rb')
  train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
  f.close()
  if len(sys.argv)>1:
    exp_name = sys.argv[1]
  else:
    exp_name = 'exp'
  os.makedirs(exp_name,exist_ok=True)
  logger = open("./"+exp_name+"/log","w")

  data = {
      'X_train': train_set[0], # training data
      'y_train': train_set[1], # training labels
      'X_val': valid_set[0],# validation data
      'y_val': valid_set[1] # validation labels
  }
  model = SoftmaxClassifier(input_dim=28*28, hidden_dim=1000, reg=0.0)
  solver = Solver(model, data, logger,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 0.02,
                    },
                    lr_decay=1,
                    num_epochs=5, batch_size=32,
                    print_every=100)
  solver.train()
  test_acc = solver.check_accuracy(test_set[0],test_set[1])
  toprint = "test_acc: "+str(test_acc)
  print(toprint)

def runCNN_multiclass():
  f = gzip.open('../mnist.pkl.gz', 'rb')
  train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
  f.close()
  exp_name = sys.argv[1]
  if len(sys.argv)>2:
    cont_exp = sys.argv[2]
  else:
    cont_exp = None
  os.makedirs(exp_name,exist_ok=True)
  logger = open("./"+exp_name+"/log","w")

  datapack = (np.concatenate((train_set[0],valid_set[0]),axis=0),
              np.concatenate((train_set[1],valid_set[1]),axis=0))

  data = {
      'X_train': datapack[0][:55000], # training data
      'y_train': datapack[1][:55000], # training labels
      'X_val': datapack[0][55000:],# validation data
      'y_val': datapack[1][55000:] # validation labels
  }

  ConvConfig = {
      'input_dim':(1, 28, 28), 
      'num_filters':16, 
      'filter_size':7,
      'hidden_dim':16, 
      'num_classes':10, 
      'weight_scale':1e-3, 
      'reg':0., 
      'bn':False, 
      'dropout':False,
      'cont_exp': cont_exp
  }

  logger.write(str(ConvConfig)+'\n')
  model = ConvNet(**ConvConfig)
  solver = Solver(model, data, logger,
                    update_rule='adam',
                    optim_config={
                      'learning_rate': 0.001,
                    },
                    lr_decay=0.9,
                    num_epochs=10, batch_size=100,
                    print_every=10,
                    exp_name=exp_name)
  solver.train()

  test_acc = solver.check_accuracy(test_set[0],test_set[1])
  toprint = "test_acc: "+str(test_acc)
  print(toprint)
  logger.write(toprint)
  logger.flush()

def main():
  runSoftmax()


if __name__ == "__main__":
  main()

