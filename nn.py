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

    def startTrain(self, training_inputs, training_outputs, training_iterations):
        
        #training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            #siphon the training data via  the neuron
            output = self.think(training_inputs)

            #computing error rate for back-propagation
            error = training_outputs - output
            
            #performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def train(self, training_iterations=10000):
        test_input = np.random.randint(0, 2, (training_iterations, 4))
        test_correct = test_input[:, 0]
        return (neural_network.think(test_input)[:, 0] >= 0.5).astype(int), test_correct

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
    neural_network.startTrain(training_inputs, training_outputs, 5000)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    output_classified, test_correct = neural_network.train()

    while(True):
    
        # Conta os acertos
        corrects = output_classified == test_correct

        # Imprime o percentual de acerto
        print("Current percentage of correct answers: %s%%" % (np.sum(corrects)/test_correct.shape[0]))

        print("\nDigit the new situation, just 1 or 0...\n")
        user_input_one = str(input("User Input One: "))
        user_input_two = str(input("User Input Two: "))
        user_input_three = str(input("User Input Three: "))
        user_input_four = str(input("User Input Four: "))

        new_situation = np.array([user_input_one, user_input_two, user_input_three, user_input_four])

        print("\nConsidering New Situation: ", user_input_one, user_input_two, user_input_three, user_input_four)

        output = (neural_network.think(new_situation) >= 0.5).astype(int)

        if int(output) == int(user_input_one):
            print("New Output data:\n" + str(output))
            print("Wow, we did it!\n")
            if input("Do you want try again? [y]yes or [n]no?\n") == 'n':
                break
        else:
            print("New Output data:\n" + str(output))
            print("Wrong output data, I need train a little bit...\n")
            output_classified, test_correct = neural_network.train()
