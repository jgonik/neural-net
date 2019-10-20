# Implement and train a neural network using NumPy
# The neural network's weights are optimized using Stochastic Gradient Descent (SGD).

from collections import OrderedDict
import numpy as np
import h5py
#data file type h5py
import time
import copy
from random import randint
import configparser
import sys

#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:,0]))
MNIST_data.close()

####################################################################################
#Neural Network Implementation

class NN:
    first_layer = {}
    second_layer = {}

    # initialize the model parameters, including the weights and offsets for each layer
    def __init__(self, inputs, hidden, outputs, activation='relu'):
        self.first_layer['weights'] = np.random.randn(hidden,inputs) / np.sqrt(num_inputs)
        self.first_layer['offsets'] = np.random.randn(hidden,1) / np.sqrt(hidden)
        self.second_layer['weights'] = np.random.randn(outputs,hidden) / np.sqrt(hidden)
        self.second_layer['offsets'] = np.random.randn(outputs,1) / np.sqrt(hidden)
        self.input_size = inputs
        self.hid_size = hidden
        self.output_size = outputs
        self.activation = activation

    # implementing various activation functions
    def __activfunc(self,Z,type = 'relu',derivative = False):
        if type == 'relu':
            if derivative == True:
                return np.array([1 if i>0 else 0 for i in np.squeeze(Z)])
            else:
                return np.array([i if i>0 else 0 for i in np.squeeze(Z)])
        elif type == 'sigmoid':
            if derivative == True:
                return 1/(1+np.exp(-Z))*(1-1/(1+np.exp(-Z)))
            else:
                return 1/(1+np.exp(-Z))
        elif type == 'tanh':
            if derivative == True:
                return 1-(np.tanh(Z))**2
            else:
                return np.tanh(Z)
        else:
            raise TypeError('Invalid type!')

    # softmax activation function
    def __softmax(self,z):
        return 1/sum(np.exp(z)) * np.exp(z)

    # cross entropy error
    def __cross_entropy_error(self,v,y):
        return -np.log(v[y])

    # forward pass through the network
    def __forward(self,x,y):
        Z = np.matmul(self.first_layer['weights'],x).reshape((self.hid_size,1)) + self.first_layer['offsets']
        H = np.array(self.__activfunc(Z, type=self.activation)).reshape((self.hid_size,1))
        U = np.matmul(self.second_layer['weights'],H).reshape((self.output_size,1)) + self.second_layer['offsets']
        predict_list = np.squeeze(self.__softmax(U))
        error = self.__cross_entropy_error(predict_list,y)
        
        layers_dict = {
            'Z':Z,
            'H':H,
            'U':U,
            'f_X':predict_list.reshape((1,self.output_size)),
            'error':error
        }
        return layers_dict

    # implement back propagation and compute gradients
    def __back_propagation(self,x,y,f_result):
        E = np.array([0]*self.output_size).reshape((1,self.output_size))
        E[0][y] = 1
        dU = (-(E - f_result['f_X'])).reshape((self.output_size,1))
        db_2 = copy.copy(dU)
        dC = np.matmul(dU,f_result['H'].transpose())
        delta = np.matmul(self.second_layer['weights'].transpose(),dU)
        db_1 = delta.reshape(self.hid_size,1)*self.__activfunc(f_result['Z'], type=self.activation, derivative=True).reshape(self.hid_size,1)
        dW = np.matmul(db_1.reshape((self.hid_size,1)),x.reshape((1,784)))

        grad = {
            'dC':dC,
            'db_2':db_2,
            'db_1':db_1,
            'dW':dW
        }
        return grad

    # update weights and offsets
    def __optimize(self,b_result, learning_rate):
        self.second_layer['weights'] -= learning_rate*b_result['dC']
        self.second_layer['offsets'] -= learning_rate*b_result['db_2']
        self.first_layer['offsets'] -= learning_rate*b_result['db_1']
        self.first_layer['weights'] -= learning_rate*b_result['dW']

    # calculate loss
    def __loss(self,X_train,Y_train):
        loss = 0
        for n in range(len(X_train)):
            y = Y_train[n]
            x = X_train[n][:]
            loss += self.__forward(x,y)['error']
        return loss

    # train the neural network
    def train(self, X_train, Y_train, num_iterations = 1000, learning_rate = 0.5, calculate_loss=False):
        # generate a random list of indices from the training set
        rand_indices = np.random.choice(len(X_train), num_iterations, replace=True)
        
        #define step size for gradient descent
        def l_rate(base_rate, ite, num_iterations, schedule = False):
        # determine whether to update step size by iteration
            if schedule == True:
                return base_rate * 10 ** (-np.floor(ite/num_iterations*5))
            else:
                return base_rate

        count = 1
        loss_dict = {}
        test_dict = {}

        for i in rand_indices:
            f_result = self.__forward(X_train[i],Y_train[i])
            b_result = self.__back_propagation(X_train[i],Y_train[i],f_result)
            self.__optimize(b_result,l_rate(learning_rate,i,num_iterations,True))
            
            if count % 1000 == 0:
                if count % 5000 == 0:
                    test = self.testing(x_test,y_test)
                    print('Trained {} times,'.format(count),'test accuracy = {}'.format(test))
                    if calculate_loss:
                        loss = self.__loss(X_train,Y_train)
                        loss_dict[str(count)]=loss
                        print('Loss =', loss)
                    test_dict[str(count)]=test
                else:
                    print('Trained {} times,'.format(count))
            count += 1

        print('Training finished!')
        return loss_dict, test_dict

    # test the model with the testing data
    def testing(self,X_test, Y_test):
        total_correct = 0
        for n in range(len(X_test)):
            y = Y_test[n]
            x = X_test[n][:]
            prediction = np.argmax(self.__forward(x,y)['f_X'])
            if (prediction == y):
                total_correct += 1
        return total_correct/np.float(len(X_test))

####################################################################################

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config_filename = sys.argv[1]
    config.read(config_filename)

    #default values
    num_inputs = 784
    activation = 'relu'
    hidden_size = 300
    num_outputs = 10
 
    for layer in config.sections():
        if layer[-1] == '0':
            num_inputs = int(config[layer]['in_size']) # define the input size
            activation = config[layer]['activation'] # define the activation function
        elif layer[-1] == '1':
            hidden_size = int(config[layer]['in_size']) # define the hidden layer size
        else:
            num_outputs = int(config[layer]['out_size']) # define the output size

    print('Number of inputs:', num_inputs)
    print('Hidden layer size:', hidden_size)
    print('Number of outputs:', num_outputs)
    print('Internal activation function:', activation)
    print('Output activation function:', 'softmax')
    # define the number of iterations
    num_iterations = int(sys.argv[2])
    # define the default step size
    learning_rate = 0.01

    # train and evaluate the neural network!
    model = NN(num_inputs,hidden_size,num_outputs, activation)
    loss_dict, test_dict = model.train(x_train,y_train,num_iterations=num_iterations,learning_rate=learning_rate)
    accuracy = model.testing(x_test,y_test)

    #save offsets to text files
    print("Saving offsets...")
    f1 = open("first_layer.txt", "a")
    first_weights = model.first_layer["weights"]
    first_offsets = model.first_layer["offsets"]
    f1.write("Weights: \n")
    for weight in first_weights:
       f1.write(str(weight) + "\n") 
    f1.write("Offsets: \n")
    for offset in first_offsets:
        f1.write(str(offset) + "\n")
    f1.close()
    f2 = open("second_layer.txt", "w")
    second_weights = model.second_layer["weights"]
    second_offsets = model.second_layer["offsets"]
    f2.write("Weights: \n")
    for weight in second_weights:
        f2.write(str(weight) + "\n")
    f2.write("Offsets: \n")
    for offset in second_offsets:
        f2.write(str(offset) + "\n")
    f2.close()
