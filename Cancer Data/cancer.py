import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

def drop_last_column(data):
    data = data.T
    data = data[:-1]
    data = data.T
    return data

url = "D:\Machine Learning github\Aplicando perceptron multicapa\Cancer Data\Cancer_Data.csv"
columnas = ['radius_mean', 
            'texture_mean', 
            'perimeter_mean', 
            'area_mean', 
            'radius_se', 
            'texture_se', 
            'perimeter_se', 
            'area_se', 
            'radius_worst', 
            'texture_worst', 
            'perimeter_worst', 
            'area_worst']

def scailing(data, minimo, maximo):
    data = (data - minimo) / (maximo - minimo)
    return data
    
def treatment_data(url, columnas):
    data = pd.read_csv(url)
    minimo = data[columnas].min()
    maximo = data[columnas].max()
    data[columnas] = scailing(data[columnas], minimo, maximo)
    data = np.array(data)
    
    train_X = data[:-69, 2:]
    train_Y = data[:-69, 1]
    train_Y = train_Y[:, np.newaxis]
    train_X = drop_last_column(train_X)
    
    test_X = data[-69:, 2:]
    test_Y = data[-69:, 1]
    test_Y = test_Y[:, np.newaxis]
    test_X = drop_last_column(test_X)
    
    return train_X.astype(float), train_Y, test_X.astype(float), test_Y

train_X, train_Y, test_X, test_Y = treatment_data(url, columnas)

# B = benign cancer         0
# M = Malignant cancer      1

def treatment_Y(data_Y):
    filas, columnas = data_Y.shape
    data_new = np.zeros((filas, columnas))
    
    for i in range(len(data_new)):
        if data_Y[i] == 'M':
            data_new[i] = 1
        else:
            data_new[i] = 0
    return data_new

train_Y_new = treatment_Y(train_Y)
test_Y_new = treatment_Y(test_Y)

class Neural_Network:
    def __init__(self, inputs, hidd1, hidd2, hidd3, output):
        self.W1 = np.random.rand(inputs, hidd1) * 2 - 1
        self.b1 = np.random.rand(1, hidd1) * 2 - 1
        self.W2 = np.random.rand(hidd1, hidd2) * 2 - 1
        self.b2 = np.random.rand(1, hidd2) * 2 - 1
        self.W3 = np.random.rand(hidd2, hidd3) * 2 - 1
        self.b3 = np.random.rand(1, hidd3) * 2 - 1
        self.W4 = np.random.rand(hidd3, output) * 2 - 1
        self.b4 = np.random.rand(1, output) * 2 - 1
        
    def sigmoid(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def dv_sig(self, z):
        dv = self.sigmoid(z) * (1 - self.sigmoid(z))
        return dv
    
    def umbral(self, z):
        return np.where(z > 0, 1, 0)
    
    def forwardPass(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = self.sigmoid(z3)
        z4 = np.dot(a3, self.W4) + self.b4
        a4 = self.umbral(z4)
        return z1, a1, z2, a2, z3, a3, z4, a4
    
    def predict(self, inputs):
        _, _, _, _, _, _, _, prediction = self.forwardPass(inputs)
        if prediction[0][0] == 1:
            return 'M'                      # Malignant cancer
        else:
            return 'B'                      # Benign cancer
    
    def update_Parameters(self, delta_W4, delta_b4, delta_W3, delta_b3, delta_W2, delta_b2, delta_W1, delta_b1, lr):
        self.W4 = self.W4 - lr * delta_W4
        self.b4 = self.b4 - lr * delta_b4
        self.W3 = self.W3 - lr * delta_W3
        self.b3 = self.b3 - lr * delta_b3
        self.W2 = self.W2 - lr * delta_W2
        self.b2 = self.b2 - lr * delta_b2
        self.W1 = self.W1 - lr * delta_W1
        self.b1 = self.b1 - lr * delta_b1
    
    def fit(self, X, Y, lr, epochs):
        loss = []
        for epoch in range(epochs):
            z1, a1, z2, a2, z3, a3, z4, output = self.forwardPass(X)
            error = (output - Y)
            error_subCapa = error @ self.W4.T * self.dv_sig(z3)
            
            delta_W4 = np.dot(a3.T, error)
            delta_b4 = np.sum(error, axis = 0, keepdims = True)
            delta_W3 = np.dot(a2.T, error_subCapa)
            delta_b3 = np.sum(error_subCapa, axis = 0, keepdims = True)
            delta_W2 = np.dot(a1.T, error_subCapa @ self.W3.T * self.dv_sig(z2))
            delta_b2 = np.sum(error_subCapa @ self.W3.T * self.dv_sig(z2), axis = 0, keepdims = True)
            delta_W1 = np.dot(X.T, error_subCapa @ self.W3.T * self.dv_sig(z2) @ self.W2.T * self.dv_sig(z1))
            delta_b1 = np.sum(error_subCapa @ self.W3.T * self.dv_sig(z2) @ self.W2.T * self.dv_sig(z1), axis = 0, keepdims = True)
            
            self.update_Parameters(delta_W4, delta_b4, delta_W3, delta_b3, delta_W2, delta_b2, delta_W1, delta_b1, lr)
            
            if epoch % 50 == 0:
                delta_error = error
                delta_error = 0.5 * (delta_error) ** 2
                mce = np.mean(abs(delta_error))
                print(f"Error {epoch}: ", mce)
                loss.append(mce)
                plt.plot(range(len(loss)), loss)
                plt.title("Error cuadratico medio")
                plt.xlabel("Iteraciones")
                plt.ylabel("MCE")
                plt.show()
                time.sleep(0.5)

p = Neural_Network(30, 20, 10, 5, 1)
prueba1 = train_X[0]
prueba1 = prueba1[:, np.newaxis]
prueba1 = prueba1
prueba1 = np.array(prueba1)
prueba1 = prueba1.T

def predicciones(data_X, data_Y, p):
    correctos = 0
    incorrectos = 0
    for i in range(len(data_X)):
        if p.predict(data_X[i]) == data_Y[i]:
            correctos += 1
        else:
            incorrectos += 1
    
    accuracy = correctos / len(data_X)
    accuracy *= 100
    print(f"Correctos: {correctos}" + "\n" +
          f"Incorrectos:  {incorrectos}" + "\n" +
          f"Accuracy: {round(accuracy, 2)}%")

predicciones(test_X, test_Y, p)
p.fit(train_X, train_Y_new, 0.001, 1000)
predicciones(test_X, test_Y, p)
paciente = np.copy(test_X)

def diagnostico(paciente):
    diag = p.predict(paciente)
    if diag == 'M':
        print("Diagnostico del paciente: Malignant cancer")
    else:
        print("Diagnostico del paciente: Benign cancer")
