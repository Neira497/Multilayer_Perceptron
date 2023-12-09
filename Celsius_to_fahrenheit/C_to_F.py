import numpy as np
import matplotlib.pyplot as plt
import time

x = np.array([[-0.40],    # -40
              [-0.10],    # 14
              [0.0],      # 32
              [0.08],      # 46
              [0.15],     # 59
              [0.22],     # 72
              [0.38]])    # 100

y = np.array([[-0.40],
              [0.14],
              [0.32],
              [0.46],
              [0.59],
              [0.72],
              [1]])

class perceptron:
    def __init__(self):
        self.W = np.random.rand(1, 1) * 2 - 1
        self.b = np.random.rand(1, 1) * 2 - 1
        
    def identify(self, z):
        return z
    
    def forwardPass(self, inputs):
        z = inputs * self.W + self.b
        a = self.identify(z)
        return z, a
    
    def prediction(self, inputs):
        inputs /= 100
        _, a = self.forwardPass(inputs)
        a = round(a[0][0], 2)
        a *= 100
        return a
    
    def fit(self, X, Y, lr, epochs):
        for epoch in range(epochs):
            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                _, output = self.forwardPass(x)
                error = (output - y)
                delta_W = error * x
                delta_b = error
                
                self.W = self.W - lr * delta_W
                self.b = self.b - lr * delta_b
                
            if epoch % 2 == 0:
                print(f"Epoca {epoch}")
                # Obtener los pesos y el bias
                w = self.W[0][0]
                b = self.b[0][0]
                
                # Calcular la línea de la regresión
                x_min = np.min(X)
                x_max = np.max(X)
                x_line = np.linspace(x_min, x_max, 100)
                y_line = w * x_line + b
                
                # Graficar la línea de la regresión
                plt.scatter(X, Y)
                plt.plot(x_line, y_line, color='red')
                plt.xlim(x_min, x_max)
                plt.show()
                time.sleep(0.3)
                
p = perceptron()
plt.scatter(x, y)
print(p.prediction(15))
p.fit(x, y, 0.4, 100)
print(p.prediction(15))