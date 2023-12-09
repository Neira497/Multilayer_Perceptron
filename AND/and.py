import numpy as np
import pandas as pd

url = "D:\Machine Learning github\Aplicando perceptron multicapa\AND\AND.csv"

def data_treatment(url):
    data = pd.read_csv(url)
    data = np.array(data)
    np.random.shuffle(data)
    
    data = data.T
    m, n = data.shape
    data_X = data[:m-1]
    data_Y = data[m-1]
    data_Y = data_Y[:, np.newaxis]
    
    return data_X.T, data_Y
    
data_X, data_Y = data_treatment(url)
test_X = np.array([[0, 0],
                   [1, 0],
                   [0, 1],
                   [1, 1]])

def predictions(test_X, p):
    for i in range(len(test_X)):
        print(f"{test_X[i]}: {round(p.predict(test_X[i]))}")
    print()
    
class Perceptron:
    def __init__(self):
        self.W = np.random.rand(2, 1) * 2 - 1
        self.b = np.random.rand(1, 1) * 2 - 1
        
    def sigmoid(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def sigmoid_deriv(self, z):
        deriv = self.sigmoid(z) * (1 - self.sigmoid(z))
        return deriv
    
    def forwardPass(self, X):
        x = np.reshape(X, (-1, 1))
        sub = np.dot(self.W.T, x)
        z = np.sum(sub) + self.b
        a = self.sigmoid(z)
        return z, a
    
    def predict(self, X):
        _, predict = self.forwardPass(X)
        return predict[0][0]
    
    def update_parameters(self, delta_W, delta_b, lr):
        self.W = self.W - lr * delta_W
        self.b = self.b - lr * delta_b
    
    def fit(self, lr, X, Y, epochs):
        for epoch in range(epochs):
            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                z, output = self.forwardPass(x)
                error = (output - y)
                error_capa = error * self.sigmoid_deriv(z)
                delta_W = x * error_capa
                delta_b = error_capa
                self.update_parameters(delta_W, delta_b, lr)
    
p = Perceptron()
print("Pre entrenaiento:")
predictions(test_X, p)
p.fit(0.5, data_X, data_Y, 1000)
print("Post entrenamiento:")
predictions(test_X, p)