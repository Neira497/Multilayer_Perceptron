import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

url = "D:\Machine Learning github\Aplicando perceptron multicapa\Aproximación sensor\Aproximación sensor.csv"

def test(voltaje):
    print(f"Voltaje: {voltaje}" + "\n" +
          f"Distancia: {p.predict(voltaje, minimo, maximo)}")

def data_treatment(url):
    data = pd.read_csv(url)
    data = np.array(data)
    np.random.shuffle(data)
    
    data = data.T
    data_X = data[:-1]
    data_Y = data[-1:]
    return data_X.T, data_Y.T

data_X, data_Y = data_treatment(url)

def scaling(data_X):
    minimo = np.min(data_X)
    maximo = np.max(data_X)
    data_X = (data_X - np.min(data_X)) / (np.max(data_X) - np.min(data_X))
    return data_X, minimo, maximo

data_X_Copy = np.copy(data_X)
data_X, minimo, maximo = scaling(data_X)

class Perceptron:
    def __init__(self, hidden):
        self.W1 = np.random.rand(1, hidden) * 2 - 1
        self.b1 = np.random.rand(1, hidden) * 2 - 1
        
        self.W2 = np.random.rand(hidden, 1) * 2 - 1
        self.b2 = np.random.rand(1, 1) * 2 - 1
        
    def linear(self, z):
        return z
    
    def sigmoid(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def deriv_sigmoid(self, z):
        deriv = self.sigmoid(z) * (1 - self.sigmoid(z))
        return deriv
    
    def forwardPass(self, inputs):
        inputs, _, _ = scaling(inputs)
        z1 = inputs @ self.W1 + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.linear(z2)
        return z1, a1, z2, a2
    
    def update_parameters(self, delta_W1, delta_b1, delta_W2, delta_b2, lr):
        self.W1 = self.W1 - lr * delta_W1
        self.b1 = self.b1 - lr * delta_b1
        self.W2 = self.W2 - lr * delta_W2
        self.b2 = self.b2 - lr * delta_b2
    
    def normalizarVoltios(self, inputs):
        normalizado = (inputs - 0.4) / (3.1 - 0.4)
        return normalizado
    
    def predict(self, inputs, minimo, maximo):
        normalizado = (inputs - minimo) / (maximo - minimo)
        z1 = normalizado @ self.W1 + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.linear(z2)
        return round(a2[0][0], 2)
    
    def ECM(self, output, y):
        ecm = (0.5 * (output - y) ** 2)
        return ecm
    
    def fit(self, lr, X, Y, stop):
        i = 0
        z1, a1, z2, output = self.forwardPass(X)
        error = (output - Y)
        
        while np.mean(abs(error)) > stop:
            i = i + 1
            z1, a1, z2, output = self.forwardPass(X)
            error = (output - Y)
            
            delta_W2 = np.dot(a1.T, error)
            delta_b2 = np.sum(error, axis = 0, keepdims = True)
            
            delta_W1 = np.dot(X.T, error @ self.W2.T * self.deriv_sigmoid(z1))
            delta_b1 = np.sum(error @ self.W2.T * self.deriv_sigmoid(z1), axis = 0, keepdims = True)
            
            self.update_parameters(delta_W1, delta_b1, delta_W2, delta_b2, lr)
            if i % 200 == 0:
              print(f"Vuelta {i}: {np.mean(abs(error))}")
              graficar(X, np.min(X), np.max(X))
              time.sleep(0.5)

def graficar(data_X_Copy, minimo, maximo):
    x = np.linspace(0, np.max(data_X_Copy), 1000)
    x = x[:, np.newaxis]
    prediccion = []
    for i in range(len(x)):
        prediccion.append(p.predict(x[i], minimo, maximo))
    prediccion = np.array(prediccion)
    prediccion = prediccion[:, np.newaxis]
    plt.scatter(data_Y, data_X_Copy)
    plt.plot(prediccion, x, color = "red")
    plt.xlim(np.min(data_Y) - 1, np.max(data_Y) + 1)
    plt.title("Sensor")
    plt.xlabel("Distancia (cm)")
    plt.ylabel("Voltaje")
    plt.scatter(p.predict(voltaje, minimo, maximo), 
                voltaje, 
                color = "green", 
                marker = "P",
                s = 300)
    plt.show()

if __name__ == "__main__":
    p = Perceptron(4)
    volts = []
    voltaje = 0.4
    volts.append(voltaje)
    voltaje = np.array(volts)
    test(volts)
    p.fit(0.001, data_X_Copy, data_Y, 0.79)
    test(volts)