import numpy as np

def test(voltaje):
    print(f"Voltaje: {voltaje}" + "\n" +
          f"Distancia: {p.predict(voltaje)}")
    
minimo = 0.4
maximo = 3.1

'''
W1: array([[-23.56514199, -32.26976488,  -6.38868974, -11.32016062]])
W2: array([[28.66895571],
       [31.23646576],
       [11.8572932 ],
       [22.97062787]])
b2: array([[4.83261629]])
b1: array([[1.27717398, 0.51359045, 4.23113758, 2.50335026]])
'''

class Perceptron:
    def __init__(self, hidden):
        self.W1 = np.array([[-23.56514199, -32.26976488,  -6.38868974, -11.32016062]])
        self.W2 = np.array([[28.66895571],
                            [31.23646576],
                            [11.8572932 ],
                            [22.97062787]])
        
        self.b1 = np.array([[4.83261629]])
        self.b2 = np.array([[1.27717398, 0.51359045, 4.23113758, 2.50335026]])
        
    def predict(self, inputs):
        normalizado = (inputs - minimo) / (maximo - minimo)
        z1 = normalizado @ self.W1 + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.linear(z2)
        return round(a2[0][0], 2)
    
    def linear(self, z):
        return z
    
    def sigmoid(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
if __name__ == "__main__":
    p = Perceptron(4)
    volts = []
    voltaje = 0.44
    volts.append(voltaje)
    voltaje = np.array(volts)
    test(voltaje)