from logistic import LogisticClassifier
from svm import SVM
from softmax import SoftmaxClassifier
from cnn import ConvNet
from solver import Solver
import pickle
import gzip




def runLogistic():
  with open('../data.pkl', "rb" ) as f:
    X,Y = pickle.load(f,encoding='latin1')

  data = {
      'X_train': X[:500], # training data
      'y_train': Y[:500], # training labels
      'X_val': X[500:750],# validation data
      'y_val': Y[500:750]# validation labels
  }
  model = LogisticClassifier(input_dim=20, hidden_dim=1000, reg=0.004)
  solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 0.01,
                    },
                    lr_decay=1,
                    num_epochs=50, batch_size=50,
                    print_every=100)
  solver.train()

def runSVM():
  with open('../data.pkl', "rb" ) as f:
    X,Y = pickle.load(f,encoding='latin1')
  # Y[Y==0] = -1
  data = {
      'X_train': X[:500], # training data
      'y_train': Y[:500], # training labels
      'X_val': X[500:750],# validation data
      'y_val': Y[500:750]# validation labels
  }
  model = SVM(input_dim=20, hidden_dim=None, reg=0.004)
  solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 0.005,
                    },
                    lr_decay=1,
                    num_epochs=100, batch_size=50,
                    print_every=100)
  solver.train()

def runSoftmax():
  # Load the dataset
  f = gzip.open('../mnist.pkl.gz', 'rb')
  train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
  f.close()

  data = {
      'X_train': train_set[0], # training data
      'y_train': train_set[1], # training labels
      'X_val': valid_set[0],# validation data
      'y_val': valid_set[1] # validation labels
  }
  model = SoftmaxClassifier(input_dim=28*28, hidden_dim=None, reg=0.004)
  solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 0.025,
                    },
                    lr_decay=0.99,
                    num_epochs=30, batch_size=128,
                    print_every=100)
  solver.train()

def runCNN_multiclass():
  f = gzip.open('../mnist.pkl.gz', 'rb')
  train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
  f.close()

  datapack = (np.concatenate((train_set[0],valid_set[0]),axis=0),
              np.concatenate((train_set[1],valid_set[1]),axis=0))

  data = {
      'X_train': datapack[0][:55000], # training data
      'y_train': datapack[1][:55000], # training labels
      'X_val': datapack[0][55000:],# validation data
      'y_val': datapack[1][55000:] # validation labels
  }
  model = ConvNet(input_dim=(1, 28, 28), num_filters=16, filter_size=7,
               hidden_dim=16, num_classes=10, weight_scale=1e-3, reg=0.)
  solver = Solver(model, data,
                    update_rule='adam',
                    optim_config={
                      'learning_rate': 0.001,
                    },
                    lr_decay=0.9,
                    num_epochs=5, batch_size=100,
                    print_every=10)
  solver.train()


def main():
  runCNN_multiclass()


if __name__ == "__main__":
  main()

