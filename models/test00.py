import numpy as np
from layers import Dense, SquaredError, GradientDescent
from model import Sequential
import matplotlib.pyplot as plt

def create_model():
    model = Sequential(output_shape=1)

    W1 = np.array([[0.4, .5], [0.4, .5], [0.4, .5]])
    W2 = np.array()



    model.add(Dense(units=1, activation="sigmoid", W=W1))

    model.compile(loss=SquaredError(), optimizer=GradientDescent(learning_rate=1))

    return model


X = np.array([[.3, .4, .5], [.5, .6, .7]])

y = np.array([[0.7, .75], [.75, .8]])


model = create_model()

print(X.shape, y.shape)
mse = SquaredError()


print("loss: ", mse.forward(model.predict(X), y))
print("---------------------------------")
pred1= mse.forward(model.predict(X), y)

dW = 1e-10
model._Sequential__layers[0]._Dense__W[0][1] += dW

pred2= mse.forward(model.predict(X), y)


model = create_model()

model.fit(X, y, epoch=1)
# print(model._Sequential__layers[0].save_a, (a2-a1)/dW, model._Sequential__layers[0].save_a - (a2-a1)/dW)
print((model._Sequential__layers[0].W_list[0] - model._Sequential__layers[0].W_list[1]))
print(np.sum(abs(pred2-pred1))/dW)

# model.fit(X, y, epoch=10000)

# plt.plot(model.save_loss)
# plt.show()

print("------------------------------- raw calculating")
sigmoid = lambda x : 1/(1+np.exp(-x))
dsigmoid = lambda x : sigmoid(x) * (1 - sigmoid(x))

MSE = lambda y_pred, y : ((y_pred-y)**2)
dMSE = lambda y_pred, y :  2*(abs(y-y_pred))

W = np.array([[0.4, .5], [0.4, .5], [0.4, .5]])
b = np.zeros((2, 2))

y_pred = sigmoid(X.dot(W)+b)
# print("loss: ", MSE(y_pred, y))

# print(dMSE(y_pred, y).shape)
# print(dsigmoid(X.dot(W)+b).shape)


print((X.T).dot(dsigmoid(X.dot(W)+b) * dMSE(y_pred, y)))