import numpy as np
import matplotlib.pyplot as plt

class Dense:
    def __init__(self, units, activation, W=None):
        self.__activation_func = activation
        self.__units = units
        self.__W = W
        self.plotting = True
        if self.plotting:
            self.W_list, self.b_list = [], []
        if self.__activation_func == "sigmoid":
            self.__activation = Sigmoid()
        elif self.__activation_func == "relu":
            self.__activation = ReLU()
        else:
            raise Exception("No activation function")


    def create_layer(self, output_shape):
        if self.__W.all() == None:
            self.__W = np.random.uniform(-0.1, 0.1, size=(self.__units, output_shape))
        self.__b = np.zeros(output_shape)
        if self.plotting == True:
            self.W_list.append(self.__W)
            self.b_list.append(self.__b)
        

    def get_shape(self):
        return self.__units
    
    def compute(self, X):
        self.X_copy = X.copy()
        self.z = X.dot(self.__W) + self.__b
        self.a = self.__activation.forward(self.z)
        return self.a


    def backpropagation(self, loss):
        self.local_save = self.__activation.backward(self.z)
        # print(self.local_save)

        self.__error = loss * self.__activation.backward(self.a)
        # print(self.X_copy.shape, self.__error.shape)
        derivative_W = np.transpose(self.X_copy).dot(self.__error)
        derivative_b = np.sum(self.__error, axis=0)

        
        return derivative_W, derivative_b

    def plot(self):
        temp = [[], []]
        for count, i in enumerate(self.W_list):
            shape = i.shape
            temp[0].append(i[np.random.randint(0, shape[0])][np.random.randint(0, shape[1])])
            temp[1].append(count)

        print(len(temp))
        print(max(temp[0]), min(temp[0]), np.mean(temp[0]))
        plt.scatter(temp[1], temp[0])
        plt.show()

    def get_error(self):
        return self.__error

    def setW(self, newW):
        self.W_list.append(newW)
        self.__W = newW
    def setb(self, newb):
        self.b_list.append(newb)
        self.__b = newb
    def getW(self):
        return self.__W
    def getb(self):
        return self.__b
            


class ReLU:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx
 

class MeanSquaredError:
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size

class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, W, b, dW, db):
        return W - self.learning_rate*dW, b - self.learning_rate*db

    