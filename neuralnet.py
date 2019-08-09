import numpy as np
from scipy.special import expit as sigmoid

class NeuralNetwork3:

    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr  = learning_rate

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activator = lambda x: sigmoid(x)

        pass


    def train(self, input_list, target_list):

        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activator(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activator(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot( output_errors * final_outputs * (1-final_outputs), np.transpose(hidden_outputs) )

        self.wih += self.lr * np.dot( hidden_errors * hidden_outputs * (1-hidden_outputs), np.transpose(inputs) )

        pass


    def query(self, input_list):

        inputs = np.array(input_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activator(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activator(final_inputs)

        return final_outputs
