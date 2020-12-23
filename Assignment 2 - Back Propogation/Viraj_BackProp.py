import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_decision_regions
from random import seed
from random import uniform

'''
Activation functions to used for hidden layer
'''
sigmoid = lambda x: 1/(1 + np.exp(-x)) 
sigmoid_prime = lambda x: sigmoid(x)*(1.0-sigmoid(x))
tanh = lambda x: np.tanh(x)
tanh_prime = lambda x: 1-np.tanh(x)**2
relu = np.vectorize(lambda x: x if x>0  else 0)
relu_prime = np.vectorize(lambda x: 1 if x>0 else 0)

classify = lambda y: 0 if y <= 0.5 else 1


'''
Accuracy Prediction function
'''
def accuracy(Y, Y_hat):
    correct = 0
    for y, y_hat in zip(Y, Y_hat):
        if y == y_hat:
            correct += 1
    return correct / len(Y)

'''
Function to add noise to the dataset
'''
def add_noise(data_points, msg):
    plt.figure()
    for i in data_points:
        i[0], i[1] = i[0] + uniform(-0.2, 0.2), i[1] + uniform(-0.2, 0.2)
        if i[2] == 1:
            plt.scatter(i[0], i[1], color = 'blue')
        else:
            plt.scatter(i[0], i[1], color = 'red')

    plt.title('{}_data with noise'.format(msg))
    plt.ylabel('Y')
    plt.xlabel('X')
    circle_1 = plt.Circle((0,0), 1, color='b', fill=False)
    rectangle_1 = plt.Rectangle((-2,-2),4,4,linewidth=1,edgecolor='r',facecolor='none')
    ax = plt.gca()
    ax.add_artist(circle_1)
    ax.add_patch(rectangle_1)
    plt.savefig("{}_noisy_data.png".format(msg))
    plt.cla()
    
    return data_points


'''
Class for MLP network
'''
class NN:
    def __init__(self, activation_fnc, epsilon=0.01, hidden_n=80):
        self.activation_fnc = activation_fnc
        self.hidden_n = hidden_n
        self.epsilon = epsilon

        if activation_fnc == "logistic":
            self.act = sigmoid
            self.act_p = sigmoid_prime
        elif activation_fnc == "hyperbolic":
            self.act = tanh
            self.act_p = tanh_prime
        elif activation_fnc == "relu":
            self.act = relu
            self.act_p = relu_prime
        
        #Set the random seed for initialising weights (1,11,...,91)
        np.random.seed(91)
        
        self.weights = {
            'W1': np.random.randn(hidden_n, 2),
            'b1': np.zeros(hidden_n),
            'W2': np.random.randn(hidden_n),
            'b2': 0,
        }
        
    '''
    Forward propoagtion function for MLP model
    '''
    def forward_propagation(self, X):    
        # this implement the vectorized equations defined above.
        Z1 = np.dot(X, self.weights['W1'].T)  + self.weights['b1']
        H = self.act(Z1)
        Z2 = np.dot(H, self.weights['W2'].T) + self.weights['b2']
        Y = sigmoid(Z2)    
        return Y, Z2, H, Z1
    
    '''
    Backward propoagtion function for MLP model
    '''
    def back_propagation(self, X, Y_T):
        N_points = X.shape[0]
        
        # forward propagation
        Y, Z2, H, Z1 = self.forward_propagation(X)
        L = (1/2) * np.sum((Y_T - Y)**2)

        # back propagation
        dLdY = -(Y_T - Y)
        dLdZ2 = np.multiply(dLdY, sigmoid_prime(Z2))
        dLdW2 = np.dot(H.T, dLdZ2)
        dLdb2 = np.dot(dLdZ2.T, np.ones(N_points))
        dLdH = np.dot(dLdZ2.reshape(N_points, 1), self.weights['W2'].reshape(1, self.hidden_n))
        dLdZ1 = np.multiply(dLdH, self.act_p(Z1))
        dLdW1 = np.dot(dLdZ1.T, X)
        dLdb1 = np.dot(dLdZ1.T, np.ones(N_points))
        
        self.gradients = {
            'W1': dLdW1,
            'b1': dLdb1,
            'W2': dLdW2,
            'b2': dLdb2,
        }
        return L
    
    '''
    Training function for MLP model
    '''
    def training(self, train_data):
        X = train_data[:, :2]
        Y = train_data[:, 2:].flatten()
        
        losses = []
        
        print("="*80)
        print("Training MLP with thresholding function: {}\n".format(self.activation_fnc))
        
        L = self.back_propagation(X, Y)
        losses.append(L)
        iteration = 1
        
        while 1:
        #Terminate condition set as when two consecutive loss values are equal till 4 decimal
        
            for weight_name in self.weights:
                self.weights[weight_name] -= self.epsilon * self.gradients[weight_name]
                
            L = self.back_propagation(X, Y)
            losses.append(L)
            iteration += 1
            
            if round(losses[-1], 4) == round(losses[-2], 4):
                print("Epoch end:  {}\nFinal loss={L:.4f}".format(iteration, L=L))
                break
            
            if iteration % 100 == 0:
                print("epoch:  {}\tlosses={L:.4f}".format(iteration, L=L))

        plt.plot(losses)
        plt.title('Loss v/s Epochs {}'.format(self.activation_fnc))
        plt.ylabel('Losses')
        plt.xlabel('Epoch')
        plt.savefig("losses_{}.png".format(self.activation_fnc))
        plt.cla()
        
    '''
    Prediction/Testing function for MLP model
    '''
    def predict(self, test_data):
        X = test_data[:, :2]
        Y = test_data[:, 2:].flatten()

        Y_, _, _, _ = self.forward_propagation(X)
        Y_hat = np.array(list(map(classify, Y_)))

        return Y_hat


if __name__ == "__main__":
    
    #Select the Activation function for hidden layer
    activation_fnc = "logistic"  #    hyperbolic logistic relu
    
    seed(1)
    np.random.seed(1)
    
    #Read training and testing data
    train_data = np.genfromtxt("100_data_points.csv", delimiter=',')
    test_data = np.genfromtxt("50_data_points.csv", delimiter=',')
    
    #Shuffle training and testing data
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    
    #Add noise to training and testing data
    train_data = add_noise(train_data, "train")
    test_data = add_noise(test_data, "test")
    
    #Separate Input and Ouput values
    X = test_data[:, :2]
    Y = test_data[:, -1]

    nn = NN(activation_fnc)
    nn.training(train_data)
    Y_hat = nn.predict(test_data)

    print("Accuracy = {}".format(accuracy(Y, Y_hat)))

    confusion_mat = confusion_matrix(Y, Y_hat)
    print("confusion_matrix:\n", confusion_mat)

    plot_decision_regions(X, Y.astype(np.int64), clf=nn)
    plt.title('Decision Boundary_{}'.format(activation_fnc))

    plt.savefig("decision_boundary_{}.png".format(activation_fnc))
