import sys
sys.path.append('neuralnet.py')
import numpy as np
import csv
import matplotlib.pyplot as plt
from neuralnet import NeuralNetwork3


##########################################

''' Training data '''

def trainer(filename):
    
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3
    number_recog_net = NeuralNetwork3(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
    training_data_list = []
    with open("mnist_data/"+filename,"rt") as f:
        data = csv.reader(f)
        for image in data:
            training_data_list.append(image)
    
    epochs = 7
    for e in range(epochs):
        for image in training_data_list:
            inputs = (np.asfarray(image[1:]) / 255 * 0.99) + 0.01
            targets = np.zeros(output_nodes) * 0.01
            targets[int(image[0])] = 0.99
            number_recog_net.train(inputs, targets)
            pass
        pass
    
    
    return number_recog_net

###########################################
        
###########################################
        
train_or_not = input("Do you wish to train the network?(Y/N): ")
if(train_or_not == "Y"):
    filename = "mnist_train.csv"
    print("Training in process...")
    number_recog_net = trainer(filename)
    print("Training Done!!")
 
       
''' Test Data '''

network_score = 0
target_score = 0
test_data_list = []

with open("mnist_data/mnist_test.csv", "rt") as f:
    data = csv.reader(f)
    for image in data:
        test_data_list.append(image)
        
for image in test_data_list:
#    plt.imshow(np.asfarray(image[1:]).reshape((28,28)), cmap='Greys', interpolation='None')
    inputs = (np.asfarray(image[1:]) / 255 * 0.99) + 0.01
    outputs = number_recog_net.query(inputs)
    network_output = np.argmax(outputs)
    actual_output = int(image[0])
    
    if(network_output == actual_output):
        network_score+=1
    else:
        target_score+=1
#    print("Actual output: ",actual_output,"\tNetwork Output: ",network_output)
    
    
print("Accuracy: ",network_score/len(test_data_list) * 100)    

#############################################
    