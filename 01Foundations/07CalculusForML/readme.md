Тооцоолол нь машин сургалтын олон алгоритмуудын амжилтын далд хөтөч юм. Бид машин сургалтын алгоритмын градиентийг оновчтой болгох хэсгийн талаар ярихад градиентийг тооцоолол ашиглан олдог.

## Application of differentiations in neural network

### Implementing back-propagation

The first thing we need to implement the activation function and the loss function. Both need to be differentiable functions or otherwise our gradient descent procedure would not work. Nowadays, it is common to use ReLU activation in the hidden layers and sigmoid activation in the output layer. We define them as a function (which assumes the input as numpy array) as well as their differentiation:

```py
import numpy as np

# Find a small float to avoid division by zero
epsilon = np.finfo(float).eps

# Sigmoid function and its differentiation
def sigmoid(z):
	return 1/(1+np.exp(-z.clip(-500, 500)))
def dsigmoid(z):
	s = sigmoid(z)
	return 2 * s * (1-s)

# Relu function and its differnetiatation
def relu(z):
	return np.maximum(0, z)
def drelu(z):
	reutn (z > 0).astype(float)
```

We deliberately clip the input of the sigmoid function to between -500 to +500 to avoid overflow. Otherwise, these functions are trivial. Then for classification, we care about accuracy but the accuracy function is not differentiable. Therefore, we use the cross entropy function as loss for training:

```py
# Loss function L(y, yhat) and its differentiation
def cross_entropy(y, yhat):
    """Binary cross entropy function
        L = - y log yhat - (1-y) log (1-yhat)

    Args:
        y, yhat (np.array): 1xn matrices which n are the number of data instances
    Returns:
        average cross entropy value of shape 1x1, averaging over the n instances
    """
    return -(y.T @ np.log(yhat.clip(epsilon)) + (1-y.T) @ np.log((1-yhat).clip(epsilon))) / y.shape[1]

def d_cross_entropy(y, yhat):
    """ dL/dyhat """
    return - np.divide(y, yhat.clip(epsilon)) + np.divide(1-y, (1-yhat).clip(epsilon))
```

In the above, we assume the output and the target variables are row matrices in numpy. Hence we use the dot product operator @ to compute the sum and divide by the number of elements in the output. Note that this design is to compute the average cross entropy over a batch of samples.

```py
class mlp:
    '''Multilayer perceptron using numpy
    '''
    def __init__(self, layersizes, activations, derivatives, lossderiv):
        """remember config, then initialize array to hold NN parameters without init"""
        # hold NN config
        self.layersizes = layersizes
        self.activations = activations
        self.derivatives = derivatives
        self.lossderiv = lossderiv
        # parameters, each is a 2D numpy array
        L = len(self.layersizes)
        self.z = [None] * L
        self.W = [None] * L
        self.b = [None] * L
        self.a = [None] * L
        self.dz = [None] * L
        self.dW = [None] * L
        self.db = [None] * L
        self.da = [None] * L

    def initialize(self, seed=42):
        np.random.seed(seed)
        sigma = 0.1
        for l, (insize, outsize) in enumerate(zip(self.layersizes, self.layersizes[1:]), 1):
            self.W[l] = np.random.randn(insize, outsize) * sigma
            self.b[l] = np.random.randn(1, outsize) * sigma

    def forward(self, x):
        self.a[0] = x
        for l, func in enumerate(self.activations, 1):
            # z = W a + b, with `a` as output from previous layer
            # `W` is of size rxs and `a` the size sxn with n the number of data instances, `z` the size rxn
            # `b` is rx1 and broadcast to each column of `z`
            self.z[l] = (self.a[l-1] @ self.W[l]) + self.b[l]
            # a = g(z), with `a` as output of this layer, of size rxn
            self.a[l] = func(self.z[l])
        return self.a[-1]
	
    	def backward(self, y, yhat):
        # first `da`, at the output
        self.da[-1] = self.lossderiv(y, yhat)
        for l, func in reversed(list(enumerate(self.derivatives, 1))):
            # compute the differentials at this layer
            self.dz[l] = self.da[l] * func(self.z[l])
            self.dW[l] = self.a[l-1].T @ self.dz[l]
            self.db[l] = np.mean(self.dz[l], axis=0, keepdims=True)
            self.da[l-1] = self.dz[l] @ self.W[l].T
 
    def update(self, eta):
        for l in range(1, len(self.W)):
            self.W[l] -= eta * self.dW[l]
            self.b[l] -= eta * self.db[l]
```
