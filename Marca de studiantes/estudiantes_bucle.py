import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url = "D:\Redes neuronales\Marca de studiantes\Student_Marks.csv"

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
    def __init__(self, layers):
        self.layers = len(layers[1])
        self.W = []
        self.b = []
        for i in range(len(layers[0]) - 1):
            self.W.append(self.parametros(layers[0][i], layers[0][i+1], True))
            self.b.append(self.parametros(1, layers[0][i+1], False))
        self.act_f = []
        for i in range(len(layers[1])):
            self.act_f.append(self.activation_function(layers[1][i]))
    
    def parametros(self, f, c, Pesos):
        if Pesos:
            return np.random.rand(f, c) * 2 - 1
        else:
            return np.random.rand(1, c) * 2 - 1
    
    def activation_function(self, funciones):
        if funciones == "sig":
            sig = (lambda x: 1 / (1 + np.exp(-x)),
                   lambda x: x * (1 - x))               # Para la activacion
            return sig
        elif funciones == "lin":
            lin = (lambda x: x,
                   lambda x: 1)
            return lin
        else:
            return "Funcion de activacion no disponible"
        
    def getW_B_shape(self):
        print(f"Tamaño de los pesos: {len(self.W)}")
        for i in range(len(self.W)):
            print(f"Tamaño del peso W{i+1}: {self.W[i].shape}")
        
        print("\n" + f"Tamaño de los bias: {len(self.b)}")
        for i in range(len(self.b)):
            print(f"Tamaño de los bias{i+1}: {self.b[i].shape}")
            
    def update_parameters(self, layer, delta_W, delta_b, lr):
        self.W[layer] = self.W[layer] - lr * (delta_W)
        self.b[layer] = self.b[layer] - lr * (delta_b)
        
    def forwardPass(self, X):
        cache = [(None, X)]
        for i in range(self.layers):
            z = np.dot(cache[i][1], self.W[i]) + self.b[i]
            a = self.act_f[i][0](z)
            cache.append((z, a))
        out = cache[-1][1]
        return out, cache
    
    def fit(self, X, Y, lr, epochs):
        delta = []
        loss = []
        for epoch in range(epochs):
            out, cache = self.forwardPass(X)
            error = (out - Y)
            
            # Error de la ultima capa
            delta.append(error * self.act_f[-1][1](out))
            for layer in reversed(range(self.layers)):
                delta_W = np.dot(cache[layer][1].T, delta[-1])
                delta_b = np.sum(delta[-1], axis = 0, keepdims = True)
                delta.append(np.dot(delta[-1], self.W[layer].T) * self.act_f[layer-1][1](cache[layer][0]))
                self.update_parameters(layer, delta_W, delta_b, lr)
            delta.reverse()
            
            if epoch % 1 == 0:
                delta_error = 0.5 * (error) ** 2
                mse = np.mean(abs(delta_error))
                print(f"Error de la epocha {epoch}: {mse}")
                loss.append(mse)
                plt.plot(range(len(loss)), loss)
            
    def predict(self, inputs):
        out, cache = self.forwardPass(inputs)
        return out[0]

layers = [[2, 5, 1],
         ["sig", "lin"]]

nn = Neural_Network(layers)
