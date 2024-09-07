import numpy as np
from layers import Dense, SquaredError, GradientDescent
from model import Sequential
import matplotlib.pyplot as plt
from layers import regularizers

def create_model():
    model = Sequential(output_shape=1)

    model.add(Dense(units=1, activation="relu", regularizer=regularizers(type="l1", l=0.1)))

    model.compile(loss=SquaredError(), optimizer=GradientDescent(learning_rate=1e-4, momentum=0.9))

    return model


X = np.random.rand(100, 1)
y = np.exp(X)/10

# print(X, y)

# plt.scatter(X, y)
# plt.show()

model = create_model()

print(X.shape, y.shape)
mse = SquaredError()


print("loss: ", np.mean(mse.forward(model.predict(X), y)))
print("---------------------------------")
pred1= mse.forward(model.predict(X), y)

d = 1e-10
model._Sequential__layers[0]._Dense__W[0][0] += d

pred2= mse.forward(model.predict(X), y)


model = create_model()

model.fit(X, y, epoch=1, batch_size=100)
# print(model._Sequential__layers[0].save_a, (a2-a1)/dW, model._Sequential__layers[0].save_a - (a2-a1)/dW)
print(model._Sequential__layers[0].W_list)
print("W1: ", (model._Sequential__layers[0].W_list[0] - model._Sequential__layers[0].W_list[1]))

print(np.sum((pred2-pred1))/d)

model = create_model()

pred1= mse.forward(model.predict(X), y)

model._Sequential__layers[0]._Dense__b[0] += d

pred2= mse.forward(model.predict(X), y)


model = create_model()

model.fit(X, y, epoch=1, batch_size=100)
# print(model._Sequential__layers[0].save_a, (a2-a1)/dW, model._Sequential__layers[0].save_a - (a2-a1)/dW)
print(model._Sequential__layers[0].b_list)
print("b1: ", (model._Sequential__layers[0].b_list[0] - model._Sequential__layers[0].b_list[1]))
print(np.sum((pred2-pred1))/d)

# model.fit(X, y, epoch=10000)

# plt.plot(model.save_loss)
# plt.show()

# print("------------------------------- raw calculating")
# sigmoid = lambda x : 1/(1+np.exp(-x))
# dsigmoid = lambda x : sigmoid(x) * (1 - sigmoid(x))

MSE = lambda y_pred, y : ((y_pred-y)**2)
# dMSE = lambda y_pred, y :  2*(abs(y-y_pred))

# W1 = np.array([[0.4, .5], [0.4, .5], [0.4, .5]])
# b1= np.zeros((1, 2))
# W2 = np.array([[.3], [.3]])
# b2 = np.zeros((1, 1))

# print("W1", W1.shape)
# print("W2", W2.shape)

# a1 = sigmoid(X.dot(W1)+b1)
# a2 = sigmoid(a1.dot(W2)+b2)

# y_pred = a2
# print("loss: ", MSE(y_pred, y))

# # print(dMSE(y_pred, y).shape)
# # print(dsigmoid(a1.dot(W2)+b2).shape)
# # print("To W2: ", a1.T.shape)

# # print(W2.T.shape)
# # print(dsigmoid(X.dot(W1)+b1).shape)
# # print("To W1: ", X.T.shape)

# # print(a1, a1.dot(W2)+b2)
# # print(W2.T)
# # print((dsigmoid(a1.dot(W2)+b2) * dMSE(y_pred, y)).dot(W2.T))

# print("dW2: ", (a1.T).dot(dsigmoid(a1.dot(W2)+b2) * dMSE(y_pred, y)))
# print("dW1: ", (X.T).dot(dsigmoid(X.dot(W1)+b1) * (dsigmoid(a1.dot(W2)+b2) * dMSE(y_pred, y)).dot(W2.T)))


print("----------------------------------- CHECKING")
model = create_model()

plt.scatter(X, model.predict(X))
plt.scatter(X, y)

plt.show()


print(np.mean(MSE(y, model.predict(X))))

model.fit(X, y, epoch=5000, batch_size=64)



print(model._Sequential__layers[0].W_list[0].shape)
lsW = []
lsb = []
for i, j in zip(model._Sequential__layers[0].W_list, model._Sequential__layers[0].b_list):
    lsW.append(i[0][0])
    try:
        lsb.append(j[0][0])
    except Exception:
        lsb.append(j[0])
    
    

print(np.mean(MSE(y, model.predict(X))))


plt.plot(lsW)
plt.plot(lsb)
plt.show()
model.plot_loss()

plt.show()
plt.scatter(X, model.predict(X))

plt.scatter(X, y)

plt.show()
