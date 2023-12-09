import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Test(X, perceptron):
    for i in range(len(X)):
        x = perceptron.predict(X[i])
        print(f"Prueba {i+1}: {x}")
        
def Test_round(X, Y, perceptron):
    for i in range(len(X)):
        x = perceptron.predict(X[i])
        x = x[0][0]
        print(f"test {i+1}: {round(x)}")
        print(f"{x} == {Y}")
        
def normalizarData(data):
    mins = data.min(axis = 0)
    maxs = data.max(axis = 0)
    data_norm = (data - mins) / (maxs - mins)
    return data_norm

def data_treatment(url):
    data = pd.read_csv(url)
    data = np.array(data)
    np.random.shuffle(data)
    
    data_X = data
    m, n = data_X.shape
    data_X = data_X.T
    data_Y = data_X[n-1]
    data_Y = data_Y[:, np.newaxis]
    data_X = data_X[:-1]
    return data_X.T, data_Y

urlTrain = "D:\Machine Learning github\Aplicando perceptron multicapa\Perro o gato\Train.csv"
urlTest = "D:\Machine Learning github\Aplicando perceptron multicapa\Perro o gato\Test.csv"

train_X, train_Y = data_treatment(urlTrain)
test_X, test_Y = data_treatment(urlTest)
train_X = normalizarData(train_X)
test_X = normalizarData(test_X)

class Perceptron:
    def __init__(self):
        self.W = np.random.rand(4, 1) * 2 - 1
        self.b = np.random.rand(1, 1) * 2 - 1
        
    def sigmoid(self, x):
        a = 1 / 1 + np.exp(-x)
        return a
    
    def deriv_sigmoid(self, x):
        a = x * (1 - x)
        return a
    
    def predict2(self, x):
        x = np.reshape(x, (-1, 1))
        n = np.dot(self.W.T, x)
        y = np.sum(n) + self.b
        a = self.sigmoid(y)
        return a
    
    def predict(self, x):
        x = np.reshape(x, (-1, 1))
        n = np.dot(self.W.T, x)
        y = np.sum(n) + self.b
        a = self.sigmoid(y)
        return a
    
    def getWeights(self):
        return self.W
    
    def getBias(self):
        return self.b
    
    def getParameters(self):
        print(f"Pesos: {self.getWeights()}" + "\n" +
              f"Bias: {self.getBias()}" + "\n")
    
    def Train(self, X, Y, lr, epochs):
        mses = []
        for epoch in range(epochs):
            error_total = 0
            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                output = self.predict(x)
                error = (output - y)
                error_total += error**2
                error_capa = error * self.deriv_sigmoid(output)
                delta_W = error_capa * x
                delta_b = error_capa
                
                self.W = self.W - (lr * delta_W)
                self.b = self.b - (lr * delta_b)
                
            if epoch % 100 == 0:
                print("Error: ", error_total)
                
            y_pred = [self.predict(x) for x in X]
            mse = MSE(y_pred, Y)
            mses.append(mse)
        plt.plot(mses)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.show()
                
def MSE(y_pred, y_true):
    return np.mean(0.5 * (y_pred - y_true)**2)
                
if __name__ == "__main__":
    prueba = np.array([1.1, 23, 3, 6])
    prueba = normalizarData(prueba)
    prueba = prueba[:, np.newaxis]
    prueba = prueba.T
    
    perceptron = Perceptron()
    y_pred = [perceptron.predict(x)[0][0] for x in test_X]
    mse = MSE(y_pred, test_Y)
    print("MSE:", mse)
    print("prueba: ",perceptron.predict(prueba))
    Test_round(prueba, 0, perceptron)
    
    perceptron.Train(train_X, train_Y, 0.4, 500)
    
    Test_round(prueba, 0, perceptron)
    print()
    print("prueba: ", perceptron.predict2(prueba))