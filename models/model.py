import numpy as np
from layers import MeanSquaredError

class Sequential:
    def __init__(self, output_shape):
        self.__layers = []
        self.__output_shape = output_shape
        self.__shape_list = []
        



    def add(self, node):
        self.__layers.append(node)
        self.__shape_list.append(node.get_shape())

    def compile(self, loss, optimizer):
        for count in range(len(self.__shape_list)):
            if count != len(self.__shape_list)-1:
                self.__layers[count].create_layer(self.__shape_list[count+1])
            else:
                self.__layers[count].create_layer(self.__output_shape)
        
        self.loss = loss
        self.optimizer = optimizer

    def predict(self, X):
        matrix = X
        for layer in range(len(self.__layers)):
            matrix = self.__layers[layer].compute(matrix)

        return matrix        

    def fit(self, X, y, epoch):
        self.save_loss = []
        
        for _ in range(epoch):
            for count, layer in enumerate(reversed(self.__layers)):
                # print(count, layer._Dense__W.shape)
                y_pred = self.predict(X)
                self.save_loss.append(MeanSquaredError().forward(y_pred, y))
                if count == 0:
                    loss = self.loss.backward(y_pred, y)
                else:
                    loss = self.__layers[count-1].get_error()
                
                dW, db = self.__layers[count].backpropagation(loss=loss)

                if count == 1:
                    self.save = dW
                    print("Shape", dW.shape, self.__layers[count].getW().shape)
                

                newW, newb = self.optimizer.update(self.__layers[count].getW(), self.__layers[count].getb(), dW, db)
                self.__layers[count].setW(newW)
                self.__layers[count].setb(newb)


        

    

        temp = self.__layers[0]
        # print(temp.plot())

    # def checkW(self, dW):
        
