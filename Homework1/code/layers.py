from builtins import range
import numpy as np


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.

    The input x has shape (N, Din) and contains a minibatch of N
    examples, where each example x[i] has shape (Din,).

    Inputs:
    - x: A numpy array containing input data, of shape (N, Din)
    - w: A numpy array of weights, of shape (Din, Dout)
    - b: A numpy array of biases, of shape (Dout,)

    Returns a tuple of:
    - out: output, of shape (N, Dout)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in out.              #
    ###########################################################################
    Xshape = x.shape
    x_flat = np.reshape(x,(x.shape[0],-1))
    out = x_flat.dot(w) + b 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    # out = np.reshape(out,Xshape[1:])
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, Dout)
    - cache: Tuple of:
      - x: Input data, of shape (N, Din)
      - w: Weights, of shape (Din, Dout)
      - b: Biases, of shape (Dout,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, Din)
    - dw: Gradient with respect to w, of shape (Din, Dout)
    - db: Gradient with respect to b, of shape (Dout,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    Xshape = x.shape
    x_flat = np.reshape(x,(x.shape[0],-1))
    dx = np.dot(dout,w.T)
    dw = x_flat.T.dot(dout)#np.matmul(x[...,None],dout[:,None,:]).sum(axis=0)
    db = np.sum(dout,axis=0)
    dx = np.reshape(dx,Xshape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = x.copy()
    out[x<=0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout.copy()
    dx[x<=0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        mu = np.mean(x,axis=0)
        var = np.var(x, axis=0)
        x_norm = (x - mu)/np.sqrt(var + eps)
        out = gamma * x_norm + beta
        running_mean = momentum*running_mean + (1-momentum)*mu
        running_var = momentum*running_var + (1-momentum)*var
        cache = (x_norm, gamma, np.sqrt(var + eps))
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_norm = (x - running_mean)/np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    x_norm, gamma, sigma = cache
    N, D = x_norm.shape

    dgamma = np.sum(dout * x_norm, axis=0)

    dbeta = np.sum(dout, axis=0)

    dx = 1/N*(gamma/sigma)*(N*dout - dbeta - x_norm*dgamma)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Implement the vanilla version of dropout.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.uniform(0,1,x.shape)# / p
        mask[mask<=p]=1
        mask[mask<1]=0
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

def conv_forward(X, W):
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter) + 1
    w_out = (w_x - w_filter) + 1

    # if not h_out.is_integer() or not w_out.is_integer():
    #     raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=0, stride=1)
    W_col = W.reshape(n_filters, -1)

    out = W_col @ X_col
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, W, X_col)

    return out, cache


def conv_backward(dout, cache):
    X, W, X_col = cache
    n_filter, d_filter, h_filter, w_filter = W.shape

    # db = np.sum(dout, axis=(0, 2, 3))
    # db = db.reshape(n_filter, -1)

    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
    dW = dout_reshaped @ X_col.T
    dW = dW.reshape(W.shape)

    W_reshape = W.reshape(n_filter, -1)
    dX_col = W_reshape.T @ dout_reshaped
    dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=0, stride=1)

    return dX, dW

# def conv_forward(x, w):
#     """
#     The input consists of N data points, each with C channels, height H and
#     width W. We convolve each input with F different filters, where each filter
#     spans all C channels and has height HH and width WW.

#     Input:
#     - x: Input data of shape (N, C, H, W)
#     - w: Filter weights of shape (F, C, HH, WW)

#     Returns a tuple of:
#     - out: Output data, of shape (N, F, H', W') where H' and W' are given by
#       H' = H - HH + 1
#       W' = W - WW + 1
#     - cache: (x, w)
#     """
#     out = None
#     ###########################################################################
#     # TODO: Implement the convolutional forward pass.                         #
#     # Hint: you can use the function np.pad for padding.                      #
#     ###########################################################################
#     N,C,H,W = x.shape
#     F,C,HH,WW = w.shape
#     H1 = H-HH+1
#     W1 = W-WW+1
#     out = np.zeros([N,F,H1,W1])
#     wn = np.tile(w,(N,1,1,1,1))
#     all_but_first = tuple(range(out.ndim))[1:]
#     for f in range(F):
#         for i in range(H1):
#             for j in range(W1):
#                 out[:,f,i,j] = np.sum(x[:,:,i:i+HH,j:j+WW] * wn[:,f], axis=all_but_first)

#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     cache = (x, w)
#     return out, cache


# def conv_backward(dout, cache):
#     """
#     Inputs:
#     - dout: Upstream derivatives.
#     - cache: A tuple of (x, w) as in conv_forward

#     Returns a tuple of:
#     - dx: Gradient with respect to x
#     - dw: Gradient with respect to w
#     """
#     dx, dw = None, None
#     ###########################################################################
#     # TODO: Implement the convolutional backward pass.                        #
#     ###########################################################################
#     x,w = cache
#     N,C,H,W = x.shape
#     F,C,HH,WW = w.shape
#     H1 = H-HH+1
#     W1 = W-WW+1
#     dx = np.zeros([N,C,H,W])
#     dw = np.zeros([F,C,HH,WW])
#     wn = np.tile(w,(N,1,1,1,1))
#     wn = wn[:,:,:,::-1,::-1]#np.flip(wn,(3,4))
#     all_but_first = tuple(range(dx.ndim))[1:]

#     dout_pad = np.zeros([N,F,H1+(HH-1)*2,W1+(WW-1)*2])
#     dout_pad[:,:,HH-1:-(HH-1),WW-1:-(WW-1)] = dout

#     for c in range(C):
#         for i in range(H):
#             for j in range(W):
#                 dx[:,c,i,j] = np.sum(dout_pad[:,:,i:i+HH,j:j+WW]*wn[:,:,c], axis=all_but_first)

#     for f in range(F):
#         for c in range(C):
#             for i in range(HH):
#                 for j in range(WW):
#                     dw[f,c,i,j] = np.sum(x[:,c,i:i+H1,j:j+W1] * dout[:,f])
    
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     return dx, dw#, db


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    (N, C, H, W) = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    HH = int(1 + (H - pool_height) / stride)
    WW = int(1 + (W - pool_width) / stride)

    out = np.zeros((N, C, HH, WW))

    for n in range(N):
        for h in range(HH):
            for w in range(WW):
                h1 = h * stride
                h2 = h1 + pool_height
                w1 = w * stride
                w2 = w1 + pool_width
                block = x[n, :, h1:h2, w1:w2]
                out[n,:,h,w] = np.max(block.reshape((C, pool_height*pool_width)), axis=1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    (x, pool_param) = cache
    (N, C, H, W) = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    HH = int(1 + (H - pool_height) / stride)
    WW = int(1 + (W - pool_width) / stride)

    dx = np.zeros_like(x)

    for n in range(N):
        for c in range(C):
            for h in range(HH):
                for w in range(WW):
                    h1 = h * stride
                    h2 = h1 + pool_height
                    w1 = w * stride
                    w2 = w1 + pool_width
                    block = np.reshape(x[n, c, h1:h2, w1:w2], (pool_height*pool_width))
                    mask = np.zeros_like(block)
                    mask[np.argmax(block)] = 1
                    dx[n,c,h1:h2,w1:w2] += np.reshape(mask,(pool_height,pool_width)) * dout[n,c,h,w]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient for binary SVM classification.
  Inputs:
  - x: Input data, of shape (N,) where x[i] is the score for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i]
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  x = np.squeeze(x)
  yt = y
  yt[y==0]=-1
  tmp = 1-yt*x
  mask = np.ones_like(tmp)
  mask[tmp<=0] = 0
  tmp = tmp*mask
  loss = np.sum(tmp)
  
  dx = -yt*mask
  dx = np.reshape(dx,[dx.shape[0],1])
  return loss, dx


def logistic_loss(x, y):
  """
  Computes the loss and gradient for binary classification with logistic 
  regression.
  Inputs:
  - x: Input data, of shape (N,) where x[i] is the logit for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i]
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  x_flat = np.squeeze(x)
  ex = np.exp(x_flat)
  loss = np.sum(-y*x_flat+np.log(1+ex))
  dx = (-y+ex/(1+ex)) 
  dx = np.reshape(dx,(len(dx),1))
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.
  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps, axis=1)[:,None]

  N = y.shape[0]
  p = softmax(x)
  log_likelihood = -np.log(p[range(N),y])
  loss = np.sum(log_likelihood) / N

  dx = p.copy()
  dx[range(N),y] -= 1
  dx = dx/N

  return loss, dx
