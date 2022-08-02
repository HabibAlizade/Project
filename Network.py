import json
import numpy as np
import pandas as pd
import random

class Network(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.n_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        self.learning_rate = 0.0 # just to save it for the good models
    def sigmoid(self, z): 
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_prime(self,z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def feedforward(self, a): 
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def update_mini_patch(self, mini_patch, eta):
        l = len(mini_patch)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_patch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w,delta_nabla_w)]
        self.biases = [b - (eta / l)* nb for b,nb in zip(self.biases, nabla_b)]        
        self.weights = [w - (eta / l) * nw for w, nw in zip(self.weights, nabla_w)]
    
    def SGD(self, training_data, epoch, mini_batch_size, eta, test_data = None): 
        self.learning_rate = eta
        n = len(training_data)
        if test_data: n_test = len(test_data)
        test_cost = []
        training_cost = []
        for j in range(epoch):
            random.shuffle(training_data)
            mini_patches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_patch in mini_patches:
                self.update_mini_patch(mini_patch, eta = eta)
            if test_data:
                if self.evaluate(test_data) > 9000:
                    self.save(f'{self.evaluate(test_data)} accuracy NN')
                print(f'Epoch {j}-th : {self.evaluate(test_data)} out of {n_test} correctly recognized')
                test_cost.append(self.total_cost(test_data))
            else: 
                print(f'Epoch {j}-th completed')
            
            training_cost.append(self.total_cost(training_data))

        return (training_cost, test_cost)
    def total_cost(self, data):
        costs = []
        for x,y in data:
            a = self.feedforward(x)
            cost = 0.5 * np.linalg.norm(a - y) ** 2    
            costs.append(cost)
        return np.sum(costs)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum([(x == y) for (x, y) in test_results])

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def backprop(self, x, y):
        activation = x
        zs = []
        activations = [activation]
        
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            activations.append(self.sigmoid(z))

        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(z[-1])    
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.n_layers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * self.sigmoid_prime(z[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        
        return (nabla_b, nabla_w)
    

    def save_network(self,filename):
        data = {
            'sizes' : self.sizes,
            'weights' : self.weights,
            'biases' : self.biases,
            'cost' : 'quadratic',
            'learning_rate' : self.learning_rate
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    def load_network(self, filename):
        with open(filename, 'r') as f: 
            data = json.load(f)
        net = Network(data['sizes'])
        net.weights = [np.array(w) for w in data['weights']]
        net.biases = [np.array(b) for b in data['biases']]
        net.learning_rate = data['learning_rete']
        return net
