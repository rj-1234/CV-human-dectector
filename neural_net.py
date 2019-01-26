from numpy import exp, array, random, dot
import numpy as np
from helper import *

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = (0.5 - np.random.random((number_of_inputs_per_neuron, number_of_neurons)) )


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    def __sigmoid(self, x):
        """
        The Sigmoid function, which describes an S shaped curve.
        """
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        """
        The derivative of the Sigmoid function.
        """
        return x * (1 - x)

    def __relu(self, x):
        """
        The Rectified Linear Units Function.
        """
        return np.maximum(0.0, x)

    def __relu_derivative(self, x):
        """
        The derivative of the Rectified Linear Units Function.
        """
        x[ x<=0 ] = 0
        x[ x>0]  = 1
        return x

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations, learning_rate = 0.01):
        helper = Helper()
        
        for iteration in range(number_of_training_iterations):
            training_set_inputs, training_set_outputs = helper.unison_shuffled_copies(training_set_inputs, training_set_outputs)
            training_set_inputs = training_set_inputs[:16]
            training_set_outputs = training_set_outputs[:16]

            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__relu_derivative(output_from_layer_1)
            
            # Calculate how much to adjust the weights by
            layer1_adjustment = learning_rate * (training_set_inputs.T.dot(layer1_delta))
            layer2_adjustment = learning_rate * (output_from_layer_1.T.dot(layer2_delta))
            
            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment
            
            if iteration % 100 == 0:
                learning_rate = learning_rate / 1.1
                print(str(iteration )+ "  Error : "+str(np.mean(self.calculate_loss(training_set_outputs, output_from_layer_2))))
    
    
    # The neural network calculates the error (Squared Error)
    def calculate_loss(self, ground_truth, predicted_output):
        return np.square(ground_truth - predicted_output)/2

    # The neural network thinks
    def think(self, inputs):
        output_from_layer1 = self.__relu(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print ("    Layer 1 Weights Shape : ")
        print (self.layer1.synaptic_weights.shape)
        print ("    Layer 2 Weights Shape :")
        print (self.layer2.synaptic_weights.shape)

    # The neural network saves its weights
    def save_weights(self):
        np.save("hidden_layer_weights.npy", self.layer1.synaptic_weights)
        np.save("output_layer_weights.npy", self.layer2.synaptic_weights)


if __name__ == "__main__":
    # load the train input and output matrices
    final_train_input = np.load("train_input.npy")
    final_train_output = np.load("train_output.npy")
    final_test_input = np.load("test_input.npy")
    final_test_output = np.load("test_output.npy")
    
    #Seed the random number generator (for reproducable results)
    random.seed(15)
    no_of_neurons_in_hidden_layer = 1000
    epochs = 201

    # Create layer 1 
    layer1 = NeuronLayer(no_of_neurons_in_hidden_layer, final_test_input[0].shape[0])

    # Create layer 2 
    layer2 = NeuronLayer(1, no_of_neurons_in_hidden_layer)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)

    # save weights
    neural_network.save_weights()

    print ("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = final_train_input
    training_set_outputs = final_train_output

    # Train the neural network using the training set.
    neural_network.train(training_set_inputs, training_set_outputs, epochs)

    print ("Stage 2) New synaptic weights after training: ")
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print ("Stage 3) Considering the test input: ")
    hidden_state, output = neural_network.think(final_test_input)
    
    final_output = []
    for i in output:
        print(i[0])
        temp = []
        if i[0] > 0.8:
            temp.append("1 : Person Detected")
        else:
            temp.append("0 : No Person Detected")
        final_output.append(temp)
    print (output, "\n",final_output,"\n" ,final_test_output)
