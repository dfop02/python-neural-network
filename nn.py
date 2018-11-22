import numpy as np

class NeuralNetwork():
    
    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)
        
        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((4, 1)) - 1

    def sigmoid(self, x):
        #applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        
        #training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            #siphon the training data via  the neuron
            output = self.think(training_inputs)

            #computing error rate for back-propagation
            error = training_outputs - output
            
            #performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def hiddenInput(self, output):
        # Hidden input with 4 inputs
        output[output > 0.5] = 1
        return output

    def think(self, inputs):
        #passing the inputs via the neuron to get output   
        #converting values to floats
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        output = self.hiddenInput(output)
        return output


if __name__ == "__main__":

    #initializing the neuron class
    neural_network = NeuralNetwork()

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    # all 16 possibilites for 4 inputs in training inputs
    training_inputs = np.array([[0,0,0,0],
                                [1,0,0,0],
                                [0,1,0,0],
                                [0,0,1,0],
                                [0,0,0,1],
                                [1,1,0,0],
                                [0,1,1,0],
                                [0,0,1,1],
                                [1,0,0,1],
                                [0,1,0,1],
                                [1,0,1,0],
                                [1,1,1,0],
                                [0,1,1,1],
                                [1,0,1,1],
                                [1,1,0,1],
                                [1,1,1,1]])

    # the anwers of all 16 possibilites (first number)
    training_outputs = np.array([[0,1,0,0,0,1,0,0,1,0,1,1,0,1,1,1]]).T

    #training taking place
    neural_network.train(training_inputs, training_outputs, 50000)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)
    while(True):
        user_input_one = str(input("User Input One: "))
        user_input_two = str(input("User Input Two: "))
        user_input_three = str(input("User Input Three: "))
        user_input_four = str(input("User Input Four: "))
        
        print("\nConsidering New Situation: ", user_input_one, user_input_two, user_input_three, user_input_four)
        output = neural_network.think(np.array([user_input_one, user_input_two, user_input_three, user_input_four]))

        if int(output) == user_input_one:
            print("New Output data:\n" + str(int(output)))
            print("Wow, we did it!\n")
        else:
            print("New Output data:\n" + str(output))
            print("Wrong output data, I need train a little bit...\n")
            neural_network.train(training_inputs, training_outputs, 50000)


