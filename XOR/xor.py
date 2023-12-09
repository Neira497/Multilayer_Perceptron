import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

url = "D:\Machine Learning github\Aplicando perceptron multicapa\XOR\XOR.csv"

def data_treatment(url):
    data = pd.read_csv(url)
    data = np.array(data)
    np.random.shuffle(data)
    
    data = data.T
    data_X = data[:-1]
    data_Y = data[-1]
    data_Y = data_Y[:, np.newaxis]
    
    return data_X.T, data_Y
    
data_X, data_Y = data_treatment(url)

class Perceptron_Simple:
    def __init__(self, inputs, hidden, outputs):
        self.W1 = np.random.rand(inputs, hidden) * 2 - 1
        self.W2 = np.random.rand(hidden, outputs) * 2 - 1
        self.b1 = np.random.rand(1, hidden) * 2 - 1
        self.b2 = np.random.rand(1, outputs) * 2 - 1
        
    def getParameters(self):
        print(f"W1: {self.W1, self.W1.shape}" + "\n" +
              f"b1: {self.b1, self.b1.shape}" + "\n" +
              f"W2: {self.W2, self.W2.shape}" + "\n" +
              f"b2: {self.b2, self.b2.shape}" + "\n")
        
    def sigmoid(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def sigmoid_deriv(self, z):
        deriv = self.sigmoid(z) * (1 - self.sigmoid(z))
        return deriv
        
    def predict(self, inputs):
        _, _, _, output = self.forwardPass(inputs)
        return output[0][0]
    
    def forwardPass(self, inputs):
        z1 = np.dot(inputs, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        return z1, a1, z2, a2
    
    def update_parameters(self, delta_W1, delta_b1, delta_W2, delta_b2, lr):
        self.W1 = self.W1 - lr * delta_W1
        self.b1 = self.b1 - lr * delta_b1
        self.W2 = self.W2 - lr * delta_W2
        self.b2 = self.b2 - lr * delta_b2
    
    def fit(self, X, Y, lr, epochs):
        loss = []
        for epoch in range(epochs):
            z1, a1, z2, output = self.forwardPass(X)
            error = (output - Y)
            error_capa = error * self.sigmoid_deriv(z2)
            
            delta_W2 = np.dot(a1.T, error_capa)
            delta_b2 = np.sum(error_capa, axis = 0, keepdims = True)
            
            delta_W1 = np.dot(X.T, (error_capa @ self.W2.T * self.sigmoid_deriv(z1)))
            delta_b1 = np.sum(error_capa @ self.W2.T * self.sigmoid_deriv(z1), axis = 0, keepdims = True)
            
            self.update_parameters(delta_W1, delta_b1, delta_W2, delta_b2, lr)
            
            if epoch % 100 == 0:
                mce = 0.5 * error ** 2
                mce = np.mean(abs(mce))
                print(mce)
                loss.append(mce)
                plt.plot(range(len(loss)), loss)
                plt.show()
                time.sleep(0.5)
            
p = Perceptron_Simple(2, 5, 1)

tests = np.array([[0, 0],
                 [1, 0],
                 [0, 1],
                 [1, 1]])

for test in range(len(tests)):
    n = p.predict(tests[test])
    n = round(n)
    print(n)

print()
p.getParameters()    
p.fit(data_X, data_Y, 0.7, 2500)
p.getParameters()
print()

for test in range(len(tests)):
    n = p.predict(tests[test])
    n = round(n)
    print(n)