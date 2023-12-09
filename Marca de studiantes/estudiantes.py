import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

url = "D:\Machine Learning github\Aplicando perceptron multicapa\Marca de studiantes\Student_Marks.csv"

def round_0_1(columna):
    data = (columna - columna.min()) / (columna.max() - columna.min())
    return data

columnas = ['number_courses', 'time_study']

def test(data_X, data_Y, nn, rango):
    correctos = 0
    incorrectos = 0
    inc = []
    
    for i in range(len(data_X)):
        if (nn.predict(data_X[i]) - rango < data_Y[i] and 
            (data_Y[i]) < nn.predict(data_X[i]) + rango):
            correctos += 1
        else:
            incorrectos += 1
            inc.append(i)
   
    accuracy = correctos / len(data_X)
    accuracy *= 100
    inc = np.array(inc)
    
    if inc.size == 0:
        print("No hay incorrectos")
    else:
        for i in range(len(inc)):
            print(f"Prueba {inc[i]}")
    print(f"Correctos: {correctos}" + "\n" +
          f"Incorrectos: {incorrectos}" + "\n" +
          f"Accuracy: {accuracy}%")

def data_treatment(url):
    data = pd.read_csv(url)
    data[columnas] = data[columnas].apply(round_0_1)
    data = np.array(data)    
    
    train_X = data[:-10, :-1]
    train_Y = data[:-10, -1]
    train_Y = train_Y[:, np.newaxis]
    
    test_X = data[-10:, :-1]
    test_Y = data[-10:, -1]
    test_Y = test_Y[:, np.newaxis]
    
    return train_X, train_Y, test_X, test_Y

train_X, train_Y, test_X, test_Y = data_treatment(url)

class Neural_Network:
    def __init__(self, inputs, hid1, hid2, hid3):
        self.W1 = np.random.rand(inputs, hid1) * 2 - 1
        self.b1 = np.random.rand(1, hid1) * 2 - 1
        self.W2 = np.random.rand(hid1, hid2) * 2 - 1
        self.b2 = np.random.rand(1, hid2) * 2 - 1
        self.W3 = np.random.rand(hid2, hid3) * 2 - 1
        self.b3 = np.random.rand(1, hid3) * 2 - 1
        
    def lineal(self, z):
        return z
    
    def sig(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def deriv_sig(self, z):
        deriv = self.sig(z) * (1 - self.sig(z))
        return deriv
    
    def forwardPass(self, X):
        z1 = X.dot(self.W1) + self.b1
        a1 = self.sig(z1)
        z2 = a1.dot(self.W2) + self.b2
        a2 = self.sig(z2)
        z3 = a2.dot(self.W3) + self.b3
        a3 = self.lineal(z3)
        return z1, a1, z2, a2, z3, a3
    
    def predict(self, inputs):
        _, _, _, _, _, output = self.forwardPass(inputs)
        return output[0][0]
    
    def update_parameters(self, delta_W1, delta_b1, delta_W2, delta_b2, delta_W3, delta_b3, lr):
        self.W1 -= (lr * delta_W1)
        self.b1 -= (lr * delta_b1)
        self.W2 -= (lr * delta_W2)
        self.b2 -= (lr * delta_b2)
        self.W3 -= (lr * delta_W3)
        self.b3 -= (lr * delta_b3)
    
    def fit(self, lr, epochs, X, Y):
        loss = []
        for epoch in range(epochs):
            z1, a1, z2, a2, z3, output = self.forwardPass(X)
            error = (output - Y)
            error_subcapa = error @ self.W3.T * self.deriv_sig(z2)
            
            delta_W3 = np.dot(a2.T, error)
            delta_b3 = np.sum(error, axis = 0, keepdims = True)
            delta_W2 = np.dot(a1.T, error_subcapa)
            delta_b2 = np.sum(error_subcapa, axis = 0, keepdims = True)
            delta_W1 = np.dot(X.T, error_subcapa @ self.W2.T * self.deriv_sig(z1))
            delta_b1 = np.sum(error_subcapa @ self.W2.T * self.deriv_sig(z1))
            
            self.update_parameters(delta_W1, delta_b1, delta_W2, delta_b2, delta_W3, delta_b3, lr)
            
            if epoch % 20 == 0:
                delta_error = error
                mce = 0.5 * (delta_error) ** 2
                mce = np.mean(abs(mce))
                loss.append(mce)
                print(f"Error {epoch}: {mce}")
                plt.plot(range(len(loss)), loss)
                plt.title("Error cuadratico medio")
                plt.xlabel("Cada 20 iteraciones")
                plt.ylabel("Error")
                plt.show()
                time.sleep(0.1)

nn = Neural_Network(2, 4, 6, 1)
test(test_X, test_Y, nn, 1.5)
nn.fit(0.0001, 2000, train_X, train_Y)
test(test_X, test_Y, nn, 1.5)