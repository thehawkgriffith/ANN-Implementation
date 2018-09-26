import numpy as np
x = [[10, 20], [14, 18]]
X = np.array(x)
y = [[1.7], [1.4]]
y = np.array(y)

def sigmoid(x, deriv = False):
    if deriv == True:
        return sigmoid(x)*(1 - sigmoid(x))
    return 1/(1 + 2.718281**(-x))

class NeuralNetwork:
    
    def __init__(self, X, hidden1, hidden2, hidden3, num_class):
        self.X = X
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.classnum = num_class
        
        self.weights = {'w1':np.random.randn(self.X.shape[1], self.hidden1),
                        'w2':np.random.randn(self.hidden1, self.hidden2),
                        'w3':np.random.randn(self.hidden2, self.hidden3),
                        'wout':np.random.randn(self.hidden3, self.classnum)}
        
        self.biases = {'b1':np.random.randn(1, self.hidden1),
                       'b2':np.random.randn(1, self.hidden2),
                       'b3':np.random.randn(1, self.hidden3),
                       'bout':np.random.randn(1, self.classnum)}
        
                                
    def predict(self):
        return self.layerout
                                
    def train(self, y, learning_rate):
        
        i = 0
        for epoch in range(100000):
            
            if i >= self.X.shape[0] - 5:
                i = 0
            
            self.layer1 = sigmoid(np.add(np.dot(self.X[i:i+5], self.weights['w1']), self.biases['b1']))
            self.layer2 = sigmoid(np.add(np.dot(self.layer1, self.weights['w2']), self.biases['b2']))
            self.layer3 = sigmoid(np.add(np.dot(self.layer2, self.weights['w3']), self.biases['b3']))    
            self.layerout = sigmoid(np.add(np.dot(self.layer3, self.weights['wout']), self.biases['bout']))
            
            errorout = self.layerout - y[i:i+5]
            delout = errorout * sigmoid(self.layerout, True)
            error3 = np.dot(delout, self.weights['wout'].T)
            del3 = error3 * sigmoid(self.layer3, True)
            error2 = np.dot(del3, self.weights['w3'].T)
            del2 = error2 * sigmoid(self.layer2, True)
            error1 = np.dot(del2, self.weights['w2'].T)
            del1 = error1 * sigmoid(self.layer1, True)
            
            self.weights['wout'] += np.dot(self.layer3.T, delout) * learning_rate
            self.weights['w3'] += np.dot(self.layer2.T, del3) * learning_rate
            self.weights['w2'] += np.dot(self.layer1.T, del2) * learning_rate
            self.weights['w1'] += np.dot(self.X[i:i+5].T, del1) * learning_rate
            
            self.biases['bout'] += np.sum(delout, axis=0) * learning_rate
            self.biases['b3'] += np.sum(del3, axis=0) * learning_rate
            self.biases['b2'] += np.sum(del2, axis=0) * learning_rate
            self.biases['b1'] += np.sum(del1, axis=0) * learning_rate
            
            if epoch%10000 == 0:
                print('Error: {}'.format(np.mean(np.abs(errorout))))

                
model = NeuralNetwork(X, 3, 3, 3, 1)
model.train(y, 0.001)
