import numpy as np

from layers import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - fc - softmax

  You may also consider adding dropout layer or batch normalization layer. 
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, bn=False, dropout=False,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.bn = bn
    self.dropout = dropout
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # conv - relu - 2x2 max pool - fc - softmax

    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters,input_dim[0],filter_size,filter_size))
    self.params['b1'] = 0#np.zeros(hidden_dim)
    self.params['W2'] = np.random.normal(0, weight_scale, (int((input_dim[1]-filter_size+1)*(input_dim[2]-filter_size+1)/4*num_filters),hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim,num_classes))
    self.params['b3'] = np.zeros(num_classes)
    if self.bn:
        self.params['gb1'] = np.random.normal(0, weight_scale,(2, int(num_filters))) # gamma beta
        self.params['gb2'] = np.random.normal(0, weight_scale,(2, hidden_dim)) # gamma beta
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # for k, v in self.params.iteritems():
    #   self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # print("forward")
    if self.bn:
        mode = 'test' if y is None else 'train'
    X = np.reshape(X,(X.shape[0],1,28,28))
    lcn,lcn_cache = conv_forward(X,W1)
     # bn
    if self.bn:
        bn_param1 = {
            'mode': mode,
            'eps': 1e-5,
            'momentum': 0.9
        }
        lcn, bn1_cache = batchnorm_forward(lcn, self.params['gb1'][0], self.params['gb1'][1], bn_param1)
    lr,lr_cache = relu_forward(lcn)
    lmx,lmx_cache = max_pool_forward(lr, pool_param)
    lmx_flat = np.reshape(lmx,[X.shape[0],-1])
    lfc,lfc_cache = fc_forward(lmx_flat,W2,b2)
    # bn
    if self.bn:
        bn_param2 = {
            'mode': mode,
            'eps': 1e-5,
            'momentum': 0.9
        }
        lfc, bn2_cache = batchnorm_forward(lfc, self.params['gb2'][0], self.params['gb2'][1], bn_param2)
    lr2,lr2_cache = relu_forward(lfc)
    if self.dropout:
        dropout_param = {
            'mode': mode,
            'p': 0.7
        }
        lr2, drp_cache = dropout_forward(lr2, dropout_param)

    lsm,lsm_cache = fc_forward(lr2,W3,b3)

    scores = lsm
    scores = np.squeeze(scores)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dx = softmax_loss(scores,y)
    # print("backward 0")
    dx,dw,db = fc_backward(dx,lsm_cache)
    grads['W3'] = dw + self.reg*self.params['W3']
    grads['b3'] = db
    if self.dropout:
        dx = dropout_backward(dx, drp_cache)
    dx = relu_backward(dx,lr2_cache)
    if self.bn:
        dx, dgamma, dbeta = batchnorm_backward(dx,bn2_cache)
        grads['gb2'] = np.concatenate((dgamma[None,:],dbeta[None,:]),axis=0)
    # print("backward 1")
    dx,dw,db = fc_backward(dx,lfc_cache)
    grads['W2'] = dw + self.reg*self.params['W2']
    grads['b2'] = db
    # print("backward 2")
    dx = np.reshape(dx,lmx.shape)
    dx = max_pool_backward(dx,lmx_cache)
    # print("backward 3")
    dx = relu_backward(dx,lr_cache)
    if self.bn:
        dx, dgamma, dbeta = batchnorm_backward(dx,bn1_cache)
        grads['gb1'] = np.concatenate((dgamma[None,:],dbeta[None,:]),axis=0)
    # print("backward 4")
    dx, dw = conv_backward(dx,lcn_cache)
    grads['W1'] = dw + self.reg*self.params['W1']
    grads['b1'] = db

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
