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
        print(f"Prueba {i+1}: {round(x)}")
        print(f"{x} == {Y[i]}")

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

urlTrain = "D:\Redes neuronales\Perro o gato\Train.csv"
urlTest = "D:\Redes neuronales\Perro o gato\Test.csv"

train_X, train_Y = data_treatment(urlTrain)
test_X, test_Y = data_treatment(urlTest)

class Perceptron:
    def __init__(self):
        self.W = np.random.rand(4, 1) * 2 - 1
        self.b = np.random.rand(1, 1) * 2 - 1
        
    def threshold(self, x):
        if x >= 0.5:
            return 1
        else:
            return 0
    
    def predict(self, x):
        x = np.reshape(x, (-1, 1))
        n = np.dot(self.W.T, x)
        y = np.sum(n) + self.b
        a = self.threshold(y)
        return np.array([[a]])  # Cambiar a una matriz de dimensiones (1,1)
    
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
            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                output = self.predict(x)
                error = (output - y)
                delta_W = error * x
                delta_b = error
                
                self.W = self.W - (lr * delta_W)
                self.b = self.b - (lr * delta_b)
                
            y_pred = [self.predict(x)[0][0] for x in X]
            mse = MSE(y_pred, Y)
            mses.append(mse)
        plt.plot(mses)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.show()
                
def MSE(y_pred, y_true):
    return np.mean(0.5 * (y_pred - y_true)**2)
                
if __name__ == "__main__":
    prueba = np.array([1, 25, 3, 8])
    prueba = prueba[:, np.newaxis]
    prueba = prueba.T
    
    perceptron = Perceptron()
    perceptron.getParameters()
    y_pred = [perceptron.predict(x) for x in test_X]
    mse = MSE(y_pred, test_Y)
    print("MSE:", mse)
    Test(test_X, perceptron)
    perceptron.Train(train_X, train_Y, 0.1, 600)
    perceptron.getParameters()
    Test(test_X, perceptron)