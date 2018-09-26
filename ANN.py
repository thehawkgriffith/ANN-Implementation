import numpy as np


def nonlin(x, deriv=False):
    if deriv == True:
        return (x*(1-x))
    return 1/(1+np.exp(-x))

class NeuralNetwork():
    
    def __init__(self, x, num_classes, inp):
        self.x = x
        self.arc = [10, 10, 10]
        self.weights = {'w0':np.random.randn(x.shape[1], 10),
                        'w1':np.random.randn(10,10),
                        'w2':np.random.randn(10,10),
                        'w3':np.random.randn(10,10),
                        'wout':np.random.randn(10, num_classes)}
        self.layer1 = np.matmul(self.x, self.weights['w0'])
        self.layer2 = np.matmul(self.layer1, self.weights['w1'])
        self.layer3 = np.matmul(self.layer2, self.weights['w2'])
        self.layer4 = np.matmul(self.layer3, self.weights['w3'])
        self.layerout = np.matmul(self.layer4, self.weights['wout'])
        
    def getOut(self):
            return self.layerout
        
    def backProp(self, y, steps):
        
        for j in range(steps):
            out_error = np.square(self.layerout - y)
            if (j%10000) == 0:
                print("Error: {}".format(str(np.mean(np.abs(out_error)))))
            layerout_delta = out_error*nonlin(self.layerout, deriv=True)
            layer4_error = np.matmul(layerout_delta, self.weights['wout'].T)
            layer4_delta = layer4_error*nonlin(self.layer4, deriv=True)
            layer3_error = np.matmul(layer4_delta, self.weights['w3'].T)
            layer3_delta = layer3_error*nonlin(self.layer3, deriv=True)
            layer2_error = np.matmul(layer3_delta, self.weights['w2'].T)
            layer2_delta = layer2_error*nonlin(self.layer2, deriv=True)
            layer1_error = np.matmul(layer2_delta, self.weights['w1'].T)
            layer1_delta = layer1_error*nonlin(self.layer1, deriv=True)
            self.weights['wout'] += np.matmul(self.layer4.T, layerout_delta)
            self.weights['w3'] += np.matmul(self.layer3.T, layer4_delta)
            self.weights['w2'] += np.matmul(self.layer2.T, layer3_delta)
            self.weights['w1'] += np.matmul(self.layer1.T, layer2_delta)
            self.weights['w0'] += np.matmul(self.x.T, layer1_delta)


X = np.array([[10,10,10], [12,132,132], [13,32,13], [14,1,13]])
model = NeuralNetwork(X, 1, 4)
model.backProp([[1], [2], [3], [4]], 60000)
