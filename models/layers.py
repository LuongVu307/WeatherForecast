import numpy as np

class ReLU:
    def __init__(self):
        pass 

    def forward(self, x):
        # Applies the ReLU activation function
        return np.maximum(0, x)
    
    def backward(self, dout):
        # Computes the gradient of the ReLU function
        return np.where(dout > 0, 1, 0)

class Sigmoid:
    def __init__(self):
        pass 

    def forward(self, x):
        # Applies the Sigmoid activation function
        out = 1 / (1 + np.exp(-x))
        return out
    
    def backward(self, dout):
        # Computes the gradient of the Sigmoid function
        dx = self.forward(dout) * (1 - self.forward(dout))
        return dx
 
class Linear:
    def __init__(self):
        pass 

    def forward(self, x):
        # Linear activation (identity function)
        out = x
        return out
    
    def backward(self, dout):
        # Gradient of the linear function
        dx = np.ones_like(dout)
        return dx

class SquaredError:
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def forward(self, y_pred, y_true):
        # Computes the squared error loss
        return ((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        # Computes the gradient of the squared error loss
        return 2 * (y_pred - y_true)

    def metrics(self, y_pred, y_true):
        # Returns the mean squared error
        return np.mean(self.forward(y_pred, y_true))

class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, W, b, dW, db):
        # Updates the weights and biases using gradient descent
        return W - self.learning_rate*dW, b - self.learning_rate*db
    
    def update_lr(self, lr):
        # Updates the learning rate
        self.learning_rate = lr
    
class regularizers:
    def __init__(self, type=None, l=0.01):
        self.type = type
        self.alpha = l

    def compute_penalty(self, ip):
        # Computes the regularization penalty
        if self.type == "l1":
            return self.alpha * np.sum(np.abs(ip))

        elif self.type == "l2":
            return self.alpha * np.sum(np.square(ip))
        
        else:
            return 0
        
    def compute_gradient(self, dout):
        # Computes the gradient of the regularization penalty
        if self.type == "l1":
            return np.sign(dout) * self.alpha
    
        elif self.type == "l2":
            return 2 * self.alpha * dout
        
        else:
            return 0


class lr_schedule:
    def __init__(self, type, lr, k, drop=None):
        self.type = type
        self.lr = lr
        self.k = k
        self.drop = drop
        if type.lower() == "step" and not drop:
            raise Exception("Step decay must have drop")


    def compute(self, epoch):
        # Computes the learning rate based on the schedule
        if self.type.lower() == "time based":
            return self.lr / (1+self.k*epoch)

        elif self.type.lower() == "exponential":
            return self.lr * np.exp(-self.k*epoch)

        elif self.type.lower() == "step":
            return self.lr * self.k**(epoch/self.drop)
        
        else:
            raise Exception("No valid type")


class Dense:
    def __init__(self, units, activation, regularizer=regularizers(), initialization="random"):
        self._activation_func = activation
        self._units = units
        self.regularizer = regularizer
        self.initialization = initialization

        # Set the activation function
        if self._activation_func == "sigmoid":
            self._activation = Sigmoid()
        elif self._activation_func == "relu":
            self._activation = ReLU()
        elif self._activation_func == "linear":
            self._activation = Linear()
        else:
            print("No activation function, ")
            self._activation = Linear()


    def create_layer(self, output_shape):
        # Initializes weights and biases based on the chosen initialization
        if self.initialization == "random":
            self._W = np.random.uniform(-0.1, 0.1, size=(self._units, output_shape))
        elif self.initialization == "zero":
            self._W = np.zeros(shape=(self._units, output_shape))
        elif self.initialization == "he":
            self._W = np.random.normal(loc=0, scale=np.sqrt(2/(self._units)), size=((self._units, output_shape)))
        elif self.initialization == "xavier":
            self._W = np.random.normal(loc=0, scale=np.sqrt(2/(self._units+output_shape)), size=((self._units, output_shape)))
        else:
            raise ValueError("Invalid Initialization")


        self._b = np.zeros(output_shape)
        

    def get_shape(self):
        # Returns the number of units in the layer
        return self._units
    
    def compute(self, X):
        # Forward pass through the layer

        self.X_copy = np.array(X)
        self.z = X.dot(self._W) + self._b
        self.a = self._activation.forward(self.z)

        self.penalty = self.regularizer.compute_penalty(self._W)

        return self.a

    def backpropagation(self, loss):
        # Backward pass through the layer

        self._error = loss * self._activation.backward(self.z)
        self.derivative_W = (np.transpose(self.X_copy)).dot(self._error) + self.regularizer.compute_gradient(self._W)
        self.derivative_b = np.sum(self._error, axis=0) + self.regularizer.compute_gradient(self._b)

        return self.derivative_W, self.derivative_b


    def get_loss(self):
        # Computes the error to be passed to the previous layer

        new_error = self._error.dot(self._W.T)
        return new_error

    def setW(self, newW):
        # Updates weights
        self._W = newW

    def setb(self, newb):
        # Updates biases
        self._b = newb

    def getW(self):
        # Returns weights
        return self._W

    def getb(self):
        # Returns biases
        return self._b

