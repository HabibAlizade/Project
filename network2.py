import json
import random
import numpy as np
import data_loader
import sys

train = data_loader.train_data
test  = data_loader.test_data
validation = data_loader.validation_data

class CrossEntropyCost(object): 
    @staticmethod
    def fct(a,y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
    
    @staticmethod
    def delta(z, a, y): 
        return (a - y) # delta = partial C / partial z
      
class QuadraticCost(object): 
    @staticmethod
    def fct(a, y):
        return 0.5 * (np.linalg.norm(a - y) ** 2)

    @staticmethod
    def delta(z, a, y): 
        return (a - y) * Network.sigmoid_prime(z) # delta = partial C / partial z
      
class Network(object): 
    def __init__(self, sizes,  cost = CrossEntropyCost()):
        self.cost = cost
        self.sizes = sizes
        self.n_layers = len(self.sizes)
    def weight_initializer(self): 
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
    
    def weight_initializer_large(self):
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b,w in zip(self.biases, self.weights): 
            a = self.sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, mini_batch_size, epoch, eta, lmbda,
        evaluation_data = False, 
        monitor_evaluation_cost = False,
        monitor_evaluation_accuracy = False,
        monitor_training_cost = False,
        monitor_training_accuracy = False):

        n = len(training_data)
#         eta_sgd = eta
        if evaluation_data : n_eval_data = len(evaluation_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []        
        n_times_eta_divided = 0

        for i in range(epoch):
            random.shuffle(training_data)
            mini_batchs = [training_data[k : k + mini_batch_size] for k in range(0,n,mini_batch_size)]

            for mini_batch in mini_batchs: 
                self.update_mini_batch(mini_batch=mini_batch, eta = eta, lmbda = lmbda, n = n)

            print(f'{i}th epoch: ', end = '')

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                if i > 0 and i % 10 == 0: 
                    x = [delta[0] - delta[1] > 0 for delta in zip(training_cost[-10:], training_cost[-11:-1])]
                    if sum(x) > 5:
                        eta = eta / 2
                        print('eta was devided')
                training_cost.append(cost)

            if monitor_training_accuracy:
                score = self.accuracy(training_data)
                training_accuracy.append(score / len(training_data))
                # print(f'{score}   ', end = '')

            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)

            if monitor_evaluation_accuracy:
                score = self.accuracy(evaluation_data)
                evaluation_accuracy.append(score / len(evaluation_data))
                print(f'{score} correctly recognized -- eta = {eta}!')

                if score / len(evaluation_data) > 0.9:
                    accuracy = score / len(evaluation_data)
                    self.save_network(f'{accuracy}% accuracy NN-model')
                    break

        return (training_cost, training_accuracy, evaluation_cost, evaluation_accuracy)
    
    
    def accuracy(self, data):
        results = [(np.argmax(self.feedforward(x)),np.argmax(y)) for x,y in data]
        return sum([int(x == y) for x,y in results])

    
    def total_cost(self, data, lmbda):
        cost = 0.0
        for x,y in data:
            a = self.feedforward(x)
            cost += self.cost.fct(a,y) / len(data)
            cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def backprop(self, x,y):
        activation = x
        activations = [activation]
        zs = []
        nabla_b = [np.zeros((y, 1)) for y in self.sizes[1:]]
        nabla_w = [np.zeros((y,x)) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = self.cost.delta(z[-1], activations[-1], y)  
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2,self.n_layers):
            sp = self.sigmoid(z[-l])
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch: 
            delta_nabla_b, delta_nabla_w = self.backprop(x,y) 
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * db for b,db in zip(self.biases,nabla_b)]
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw for w,nw in zip(self.weights, nabla_w)]
    

    def save_network(self, filename):
        data = {
            'sizes' : self.sizes,
            'weights' :[w.tolist() for w in self.weights],
            'biases' : [b.tolist() for b in self.biases],
            'cost' : str(self.cost.__name__),
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    def load_network(self,filename):
        with open(filename, 'r') as f: 
            data = json.load(f)
        cost = getattr(sys.modules[__name__], data["cost"])
        net = Network(data['sizes'], cost = cost)
        net.weights = [np.array(w) for w in data['weights']]
        net.biases = [np.array(b) for b in data['biases']]
        return net
    
    def sigmoid(self, z): 
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
