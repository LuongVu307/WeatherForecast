import numpy as np
import matplotlib.pyplot as plt
import pickle
from models.preprocessing import Pipeline, pipe0

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
        matrix = np.array(X)
        for layer in range(len(self.__layers)):
            matrix = self.__layers[layer].compute(matrix)
        return matrix        

    def fit(self, X, y, epoch=50, batch_size=1, validation=None, patient=None, lr_schedule=None):
        self.training_loss = []
        self.validation_loss = []
        if patient and validation:
            count_patient = 0
        
        for _ in range(epoch):

            if lr_schedule:
                new_lr = lr_schedule.compute(_+1)
                self.optimizer.update_lr(new_lr)

            update = []
            rand = np.random.choice(X.shape[0], batch_size, replace=True)
            X_batch = X[rand]
            y_batch = y[rand]
            y_pred = self.predict(X_batch)
            # print("PRECOMPUTE")
            # for layer in reversed(range(len(self.__layers))):
                # print(layer, end="   -")
                # print(self.__layers[layer]._Dense__W.shape, end="/ ")
                # print(self.__layers[layer].X_copy.shape, self.__layers[layer].z.shape, self.__layers[layer].a.shape)
            for count in reversed(range(len(self.__layers))):
                # print("count", count)
                # print("W shape: ", self.__layers[count]._Dense__W.shape)
                # print(self.__layers[count]._Dense__W)
                if count == len(self.__layers)-1:
                    loss = self.loss.backward(y_pred, y_batch)
                    saving_loss = round(self.loss.metrics(y_pred, y_batch), 7)
                    self.training_loss.append(self.loss.metrics(y_pred, y_batch))
                else:
                    loss = self.__layers[count+1].get_loss()
                
                # print("TRAING_SHAPE: ", self.__layers[count].X_copy.shape, self.__layers[count].z.shape, self.__layers[count].a.shape)
                dW, db = self.__layers[count].backpropagation(loss=loss)
                update.append((dW, db))
                

                    # print("Shape", dW.shape, self.__layers[count].getW().shape)
            for dall, count in zip(update, reversed(range(len(self.__layers)))):
                dW, db = dall[0], dall[1]
                newW, newb = self.optimizer.update(self.__layers[count].getW(), self.__layers[count].getb(), dW, db)
                self.__layers[count].setW(newW)
                self.__layers[count].setb(newb)

            if validation:
                val_pred = self.predict(validation[0])
                metrics = round(self.loss.metrics(validation[1], val_pred), 7)
                if patient and len(self.validation_loss) != 0:
                    direction = abs(self.validation_loss[-1]-metrics)/(self.validation_loss[-1]-metrics)
                    if direction == -1:
                        if count_patient != patient:
                            count_patient += 1
                        else:
                            print("Early Stopping")
                            break
                    else:
                        count_patient = 0
                self.validation_loss.append(self.loss.metrics(validation[1], val_pred))
                
            else:
                metrics = None
                

            print(f"Epoch {_+1} - Training Loss: {saving_loss} - Validation Loss: {metrics} - Learning Rate: {self.optimizer.learning_rate}")

        # print(temp.plot())

    # def checkW(self, dW):

    def plot_loss(self):
        # print(len(self.training_loss))
        plt.plot(np.arange(len(self.training_loss)), self.training_loss)
        plt.plot(np.arange(len(self.validation_loss)), self.validation_loss)
        plt.show()
    
    def plot_W(self):
        length = len(self.__layers)
        plt.figure(figsize=(15, 15))
        for i in range(len(self.__layers)):
            plt.subplot(int(np.sqrt(length)), (length//int(np.sqrt(length)))+1 , i+1)
            W = self.__layers[i].getW().flatten()
            
            plt.hist(W, bins=50, edgecolor='black')
            plt.title(f'Histogram of Weights in Layer {i+1} ')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')

    def plot_b(self):
        length = len(self.__layers)
        plt.figure(figsize=(15, 15))
        for i in range(len(self.__layers)):
            plt.subplot(int(np.sqrt(length)), (length//int(np.sqrt(length)))+1 , i+1)
            b = self.__layers[i].getb().flatten()
            
            plt.hist(b, bins=50, edgecolor='black')
            plt.title(f'Histogram of bias in Layer {i+1} ')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')


def save_model(model,  name="model0"):
    with open(f'save\\{name}.pkl', 'wb') as file:
        pickle.dump(model, file)

def load_model(name):
    with open(f'save\\{name}.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    return loaded_model
        