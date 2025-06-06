import unittest
import tensorflow as tf


from model import *
from layers import *


class TestC1(unittest.TestCase):

    def setUp(self):
        self.model = Sequential(output_shape=1)
        self.X = np.random.rand(2, 2)
        self.y = np.random.rand(2, 1)
        self.activation_choices = ["ReLU", "Sigmoid", "linear"]
        self.a1, self.a2 = np.random.choice(self.activation_choices), np.random.choice(self.activation_choices)
        self.rand_layer = np.random.randint(1, 50)


    def test_add_layer_with_activation(self):
        for activation in self.activation_choices:
            self.model.add(Dense(self.rand_layer, activation=activation))
            self.assertEqual(self.model._layers[0]._units, self.rand_layer)
    
    def test_add_layer_without_activation(self):
        with self.assertRaises(TypeError):
            self.model.add(Dense(self.rand_layer))

    def test_add_layer_with_regularization(self):
        self.model.add(Dense(self.rand_layer, "ReLU", regularizer=regularizers(type="l1")))
        self.assertEqual(self.model._layers[0].regularizer.type, "l1")

    def test_predict(self):
        self.model.add(Dense(2, self.a1))
        self.model.add(Dense(1, self.a2))
        self.model.compile(loss=SquaredError(), optimizer=GradientDescent(learning_rate=1))
        self.model.predict(self.X)

    def test_train(self):
        self.model.add(Dense(2, self.a1))
        self.model.add(Dense(1, self.a2))
        self.model.compile(loss=SquaredError(), optimizer=GradientDescent(learning_rate=1))

        self.model.fit(self.X, self.y, epoch=1)

class TestC2(unittest.TestCase):

    def setUp(self):

        self.data_size = np.random.randint(1, 20)
        self.input_shape = np.random.randint(1, 10)
        self.output_shape = np.random.randint(1, 5)
        self.activation_choices = ["ReLU", "Sigmoid", "linear"]

        self.model = Sequential(output_shape=self.output_shape)
        self.X = np.random.rand(self.data_size, self.input_shape)
        self.y = np.random.rand(self.data_size, self.output_shape)

        self.model.add(Dense(self.input_shape, activation=np.random.choice(self.activation_choices)))

    def test1(self):
        for _ in range(0, 5):
            self.model.add(Dense(np.random.randint(1, 20), activation=np.random.choice(self.activation_choices)))
                           
        self.model.add(Dense(self.output_shape, activation=np.random.choice(self.activation_choices)))

        self.model.compile(SquaredError(), GradientDescent(learning_rate=1))


    def test2(self):

        for _ in range(0, 5):
            self.model.add(Dense(np.random.randint(1, 20), activation=np.random.choice(self.activation_choices)))
                           
        self.model.add(Dense(self.output_shape, activation=np.random.choice(self.activation_choices)))

        self.model.compile(SquaredError(), GradientDescent(learning_rate=1))

    


class TestC3(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.data_size = np.random.randint(1, 6)
        self.input_shape = np.random.randint(1, 6)
        self.output_shape = np.random.randint(1, 4)
        self.learning_rate = np.random.randint(1, 100)/1000
        self.activation_choices = ["relu", "sigmoid", "linear"]

        self.model = Sequential(output_shape=self.output_shape)
        self.X = np.random.rand(self.data_size, self.input_shape)
        self.y = np.random.rand(self.data_size, self.output_shape)
        self.a1, self.a2 = np.random.choice(self.activation_choices), np.random.choice(self.activation_choices) 
        self.model.add(Dense(self.input_shape, activation=self.a1))
                           
        self.model.add(Dense(self.output_shape, activation=self.a2))

        self.model.compile(SquaredError(), GradientDescent(learning_rate=self.learning_rate))


    def test_predict(self):
        y_pred = self.model.predict(self.X)
        # print(self.input_shape, self.output_shape, self.data_size)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.output_shape, activation=self.a1, input_shape=(None, self.input_shape)),
            tf.keras.layers.Dense(self.output_shape, activation=self.a2)
        ])
        # print(self.model._layers[1]._W.shape, model.layers[1].get_weights()[0].shape)

        model.layers[0].set_weights((self.model._layers[0]._W, self.model._layers[0]._b))
        model.layers[1].set_weights((self.model._layers[1]._W, self.model._layers[1]._b))

        # print(model.predict(self.X), model.predict(self.X).shape)
        # print(self.model.predict(self.X), self.model.predict(self.X).shape)

        np.testing.assert_allclose(y_pred, model.predict(self.X), rtol=1e-5,atol=1e-8)

        

    
    def test_train(self):
        layer = np.random.randint(0, 2)
        size = self.model._layers[layer]._W.shape
        weights = (np.random.randint(0, size[0]), np.random.randint(0, size[1])) 
        # layer = 0
        # weights = (0, 0)
        small = 1e-6
        # print(self.data_size, self.input_shape, self.output_shape)

        # print(f"Weights at layer {layer}, position {weights}")

        loss1 = np.sum(np.abs(SquaredError().forward(self.model.predict(self.X), self.y)))
        self.model._layers[layer]._W[weights[0]][weights[1]] += small

        loss2 = np.sum(np.abs(SquaredError().forward(self.model.predict(self.X), self.y)))

        dW = (loss2-loss1)/small
        # print(f"Derivative: {dW}")

        self.model._layers[layer]._W[weights[0]][weights[1]] -= small
        first_W = self.model._layers[layer]._W[weights[0]][weights[1]]

        self.model.fit(self.X, self.y, epoch=1, batch_size=self.data_size)

        second_W = self.model._layers[layer]._W[weights[0]][weights[1]]


        # for i in range(len(self.model._layers)):
        #     print(f"layer {i} - Calculated from training: ", self.model._layers[i].derivative_W)

        # print(self.model.training_loss)
        # print(dW, self.model._layers[layer].derivative_W[weights[0]][weights[1]])
        np.testing.assert_allclose(dW, (first_W-second_W)/self.learning_rate, rtol=1e-5,atol=1e-8)

            

    def test_activation(self):
        X = np.linspace(-10, 10, 100) + np.random.uniform(-0.01, 0.01, 100)
        np.testing.assert_allclose(Linear().forward(X), np.array(tf.keras.activations.linear(X)), rtol=1e-5,atol=1e-8)
        np.testing.assert_allclose(ReLU().forward(X), np.array(tf.keras.activations.relu(X)), rtol=1e-5,atol=1e-8)
        np.testing.assert_allclose(Sigmoid().forward(X), np.array(tf.keras.activations.sigmoid(X)), rtol=1e-5,atol=1e-8)

    def test_regularization(self):
        X = np.random.normal(0.05, 0.01, 100)
        # np.testing.assert_allclose(regularizers(type="l1", l=0.01).compute_penalty(X), np.array(tf.keras.regularizers.l1(0.01)(X)), rtol=1e-5,atol=1e-8)
        alpha = np.random.randint(1, 100)/1000
        np.testing.assert_allclose(regularizers(type="l1", l=alpha).compute_penalty(X), np.array(tf.keras.regularizers.l1(alpha)(X)), rtol=1e-5,atol=1e-8)
        np.testing.assert_allclose(regularizers(type="l2", l=alpha).compute_penalty(X), np.array(tf.keras.regularizers.l2(alpha)(X)), rtol=1e-5,atol=1e-8)


    def test_loss_function(self):
        y_pred = np.random.normal(0.05, 0.01, 100)
        y_true = np.random.normal(0.05, 0.01, 100)

        np.testing.assert_allclose(SquaredError().metrics(y_pred, y_true), tf.keras.losses.mean_squared_error(y_true, y_pred), rtol=1e-5,atol=1e-8)




if __name__ == "__main__":
    unittest.main()
