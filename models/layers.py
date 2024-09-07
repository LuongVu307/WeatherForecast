import numpy as np
import matplotlib.pyplot as plt         

class ReLU:
    def __init__(self):
        pass 

    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, dout):
        return np.where(dout > 0, 1, 0)


class Sigmoid:
    def __init__(self):
        pass 

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        return out
    
    def backward(self, dout):
        dx = self.forward(dout) * (1 - self.forward(dout))
        return dx
 
class Linear:
    def __init__(self):
        pass 

    def forward(self, x):
        out = x
        return out
    
    def backward(self, dout):
        dx = np.ones_like(dout)
        return dx

class SquaredError:
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def forward(self, y_pred, y_true):
        return ((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true)

    def metrics(self, y_pred, y_true):
        return np.mean(self.forward(y_pred, y_true))

class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, W, b, dW, db):
        # print("W", W,  self.learning_rate*dW, W - self.learning_rate*dW)
        # print("Dw db: ", self.v_db, self.v_dW)
        return W - self.learning_rate*dW, b - self.learning_rate*db
    
    def update_lr(self, lr):
        self.learning_rate = lr
    
    

class regularizers:
    def __init__(self, type="None", l=0.01):
        self.type = type
        self.alpha = l

    def compute_penalty(self, ip):
        if type == "l1":
            return self.alpha * np.sum(np.abs(ip))

        elif type == "l2":
            return self.alpha * 1/2 * np.sum(np.square(ip))
        
        else:
            return 0
        
    def compute_gradient(self, ip):
        if type == "l1":
            return np.sign(ip) * self.l
    
        elif type == "l2":
            return 2 * self.l * ip
        
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
        self.__activation_func = activation
        self.__units = units
        self.regularizer = regularizer
        self.plotting = True
        self.initialization = initialization
        # if self.plotting:
        #     self.W_list, self.b_list = [], []
        if self.__activation_func == "sigmoid":
            self.__activation = Sigmoid()
        elif self.__activation_func == "relu":
            self.__activation = ReLU()
        elif self.__activation_func == "linear":
            self.__activation = Linear()
        else:
            print("No activation function, converting into default activation function: Linear")
            self.__activation = Linear()


    def create_layer(self, output_shape):
        if self.initialization == "random":
            self.__W = np.random.uniform(-0.1, 0.1, size=(self.__units, output_shape))
        elif self.initialization == "zero":
            self.__W = np.zeros(shape=(self.__units, output_shape))
        elif self.initialization == "he":
            self.__W = np.random.normal(loc=0, scale=np.sqrt(2/(self.__units)), size=((self.__units, output_shape)))
        elif self.initialization == "xavier":
            self.__W = np.random.normal(loc=0, scale=np.sqrt(2/(self.__units+output_shape)), size=((self.__units, output_shape)))
        else:
            raise ValueError("Invalid Initialization")


        self.__b = np.zeros(output_shape)

        # if self.plotting == True:
        #     self.W_list.append(self.__W)
        #     self.b_list.append(self.__b)
        

    def get_shape(self):
        return self.__units
    
    def compute(self, X):

        self.X_copy = np.array(X)
        self.z = X.dot(self.__W) + self.__b
        self.a = self.__activation.forward(self.z)
        return self.a


    def backpropagation(self, loss):
        self.__error = loss * self.__activation.backward(self.z)
        # print("computing shape: ", self.X_copy.shape, self.z.shape, self.a.shape)
        # print("SHAPE: ", loss.shape, self.__activation.backward(self.a).shape, self.X_copy.T.shape)

        # print(self.z, "-", self.a)
        # print(self.__activation.backward(self.z), loss)
        # print("z", self.z)
        # print("X", self.X_copy)

        # print(self.__error)

        derivative_W = (np.transpose(self.X_copy)).dot(self.__error) + self.regularizer.compute_gradient(self.__W)
        derivative_b = np.sum(self.__error, axis=0) + self.regularizer.compute_gradient(self.__b)

        # print(derivative_W.shape)

        # print(self.__W)
        return derivative_W, derivative_b

    # def plot(self):
    #     temp = [[], []]
    #     for count, i in enumerate(self.W_list):
    #         shape = i.shape
    #         temp[0].append(i[np.random.randint(0, shape[0])][np.random.randint(0, shape[1])])
    #         temp[1].append(count)

    #     print(len(temp))
    #     print(max(temp[0]), min(temp[0]), np.mean(temp[0]))
    #     plt.scatter(temp[1], temp[0])
    #     plt.show()

    def get_loss(self):
        # print(self.__W.T)
        # print("GETLOSS: ", self.__error.dot(self.__W.T))
        return self.__error.dot(self.__W.T)

    def setW(self, newW):
        # self.W_list.append(newW)
        self.__W = newW
    def setb(self, newb):
        # self.b_list.append(newb)
        self.__b = newb
    def getW(self):
        return self.__W
    def getb(self):
        return self.__b

