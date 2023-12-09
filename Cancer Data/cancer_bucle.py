import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

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
    np.random.shuffle(data)
    
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
    def __init__(self, layers):
        self.layers = len(layers[1])
        self.W = []
        self.b = []
        for i in range(self.layers):
            self.W.append(self.parametros(layers[0][i], layers[0][i+1], True))
            self.b.append(self.parametros(1, layers[0][i+1], False))
        self.act_f = []
        for i in range(self.layers):
            self.act_f.append(self.activation_function(layers[1][i]))
    
    def parametros(self, f, c, Pesos):
        if Pesos:
            return np.random.rand(f, c) * 2 - 1
        else:
            return np.random.rand(1, c) * 2 - 1
    
    def activation_function(self, funciones):
        if funciones == "sig":
            sig = (lambda x: 1 / (1 + np.exp(-x)),
                   lambda x: sig[0](x) * (1 - sig[0](x)))
            return sig
        elif funciones == "lin":
            lin = (lambda x: x,
                   lambda x: 1)
            return lin
        elif funciones == "umb":
            umb = (lambda x: np.where(x > 0, 1, 0),
                   lambda x: 1)
            return umb
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
        z_out = cache[-1][0]
        a_out = cache[-1][1]
        return z_out, a_out, cache
    
    def fit(self, X, Y, lr, epochs):
        delta = []
        loss = []
        
        for epoch in range(epochs):
            z_out, a_out, cache = self.forwardPass(X)
            error = (a_out - Y)
            
            # Error de la ultima capa
            delta.append(error * self.act_f[-1][1](z_out))
            for layer in reversed(range(self.layers)):
                delta_W = np.dot(cache[layer][1].T, delta[-1])
                delta_b = np.sum(delta[-1], axis = 0, keepdims = True)
                delta.append(np.dot(delta[-1], self.W[layer].T) * self.act_f[layer-1][1](cache[layer][0]))
                self.update_parameters(layer, delta_W, delta_b, lr)

            delta = []
            
            if epoch % 50 == 0:
                delta_error = 0.5 * (error) ** 2
                mse = np.mean(abs(delta_error))
                # print(f"Error de la epocha {epoch}: {mse}")
                loss.append(mse)
                plt.plot(range(len(loss)), loss)
                plt.title("Error cuadratico medio")
                plt.xlabel("Iteraciones")
                plt.ylabel("MCE")
                plt.show()
                time.sleep(0.5)
                
            if epoch+1 == epochs:
                print("\n" + "After processing" + "\n")
            
    def predict(self, inputs):
        z_out, a_out, cache = self.forwardPass(inputs)
        return a_out[0][0]

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
    print(f"Diagnosticos correctos: {correctos}" + "\n" +
          f"Diagnosticos incorrectos:  {incorrectos}" + "\n" +
          f"Accuracy: {round(accuracy, 2)}%")
    
layers = [[30, 20, 10, 5, 1],
         ["sig", "umb", "sig", "umb"]]

nn = Neural_Network(layers)
_, _, cache = nn.forwardPass(train_X)
predicciones(test_X, test_Y_new, nn)
nn.fit(train_X, train_Y_new, 0.001, 1000)
predicciones(test_X, test_Y_new, nn)
paciente = np.copy(test_X)

_, _, cache = nn.forwardPass(test_X)
def diagnostico(paciente):
    diag = nn.predict(paciente)
    
    if diag == 'M':
        print("Diagnostico del paciente: Malignant cancer")
    else:
        print("Diagnostico del paciente: Benign cancer")