import numpy as np
import matplotlib.pyplot as plt
import pickle

# Class to define and train a simple neural network model
class Sequential:
    def __init__(self, output_shape):
        # Initialize model with an empty list of layers, output shape, and an empty shape list
        self._layers = []
        self.__output_shape = output_shape
        self.__shape_list = []

    def add(self, node):
        # Add a layer (node) to the model
        self._layers.append(node)
        self.__shape_list.append(node.get_shape())

    def compile(self, loss, optimizer):
        # Initialize layers with the appropriate shapes
        for count in range(len(self.__shape_list)):
            if count != len(self.__shape_list) - 1:
                self._layers[count].create_layer(self.__shape_list[count + 1])
            else:
                self._layers[count].create_layer(self.__output_shape)
        # Store loss function and optimizer
        self.loss = loss
        self.optimizer = optimizer

    def predict(self, X):
        # Perform forward pass through all layers and return the output
        output = np.array(X)
        self.penalty = 0
        for layer in range(len(self._layers)):
            output = self._layers[layer].compute(output)
            # print(output)
            self.penalty += self._layers[layer].penalty
        return output

    def fit(self, X, y, epoch=50, batch_size=1, validation=None, patient=None, lr_schedule=None):
        # Store training and validation loss for each epoch
        self.training_loss = []
        self.validation_loss = []
        if patient and validation:
            count_patient = 0

        # Loop over the number of epochs
        for _ in range(epoch):

            # If learning rate schedule is provided, update learning rate
            if lr_schedule:
                new_lr = lr_schedule.compute(_ + 1)
                self.optimizer.update_lr(new_lr)

            update = []
            rand = np.random.choice(X.shape[0], batch_size, replace=False)
            X_batch = X[rand]
            y_batch = y[rand]
            y_pred = self.predict(X_batch)

            # Backpropagate through each layer and calculate gradients
            for count in reversed(range(len(self._layers))):
                if count == len(self._layers) - 1:
                    loss = self.loss.backward(y_pred, y_batch) + self.penalty
                    saving_loss = round(self.loss.metrics(y_pred, y_batch), 7)
                    self.training_loss.append(self.loss.metrics(y_pred, y_batch))
                else:
                    loss = self._layers[count + 1].get_loss()

                # Compute gradients for weights and biases in each layer
                dW, db = self._layers[count].backpropagation(loss=loss)
                update.append((dW, db))

            # Update weights and biases using the optimizer
            for dall, count in zip(update, reversed(range(len(self._layers)))):  # Iterate through layers in reverse
                dW, db = dall[0], dall[1]
                newW, newb = self.optimizer.update(
                    self._layers[count].getW(), self._layers[count].getb(), dW, db
                )
                self._layers[count].setW(newW)
                self._layers[count].setb(newb)

            # If validation data is provided, calculate validation loss
            if validation:
                val_pred = self.predict(validation[0])
                metrics = round(self.loss.metrics(validation[1], val_pred), 7)
                if patient and len(self.validation_loss) != 0:
                    direction = abs(self.validation_loss[-1] - metrics) / (self.validation_loss[-1] - metrics)
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

            # Print progress at the end of each epoch
            print(f"Epoch {_ + 1} - Training Loss: {saving_loss} - Validation Loss: {metrics} - Learning Rate: {self.optimizer.learning_rate}")

    def plot_loss(self):
        # Plot training and validation loss over epochs
        plt.plot(np.arange(len(self.training_loss)), self.training_loss, label='Training Loss')
        plt.plot(np.arange(len(self.validation_loss)), self.validation_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_W(self):
        # Plot histograms of weights in each layer
        length = len(self._layers)
        plt.figure(figsize=(15, 15))
        for i in range(len(self._layers)):
            plt.subplot(
                int(np.sqrt(length)), (length // int(np.sqrt(length))) + 1, i + 1
            )
            W = self._layers[i].getW().flatten()  # Flatten weights for visualization
            plt.hist(W, bins=50, edgecolor='black')
            plt.title(f'Histogram of Weights in Layer {i + 1}')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
        plt.show()

    def plot_b(self):
        # Plot histograms of biases in each layer
        length = len(self._layers)
        plt.figure(figsize=(15, 15))
        for i in range(len(self._layers)):
            plt.subplot(
                int(np.sqrt(length)), (length // int(np.sqrt(length))) + 1, i + 1
            )
            b = self._layers[i].getb().flatten()  # Flatten biases for visualization
            plt.hist(b, bins=50, edgecolor='black')
            plt.title(f'Histogram of Biases in Layer {i + 1}')
            plt.xlabel('Bias Value')
            plt.ylabel('Frequency')
        plt.show()


# Function to save the trained model to a file
def save_model(model, name="model0"):
    with open(f'save\\{name}.pkl', 'wb') as file:
        pickle.dump(model, file)

# Function to load a trained model from a file
def load_model(name):
    with open(f'save\\{name}.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model
