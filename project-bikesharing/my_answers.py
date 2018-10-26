import numpy as np
import pandas as pd

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        #self.activation_function = lambda x: 1. / (1 + np.exp(-x))
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        self.activation_function = sigmoid

        self.W1, self.W2 = None, None
        self.h1, self.a1, self.h2, self.a2 = None, None, None, None

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        #print("train")
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # NOTE: ignoring DLND notation / process for cs231n-esque.
        self.W1 = self.weights_input_to_hidden
        self.W2 = self.weights_hidden_to_output
        x = 1*X if len(X.shape) > 1 else X[None, :]

        #print("W1", self.W1.shape)
        #print("W2", self.W2.shape)
        #print("x", x.shape)
        self.h1 = x.dot(self.W1)
        #print("h1", self.h1.shape)
        self.a1 = self.activation_function(self.h1)
        #print("a1", self.a1.shape)
        self.h2 = self.a1.dot(self.W2)
        #print("h2", self.h2.shape)
        self.a2 = 1*self.h2
        #print("a2", self.a2.shape)
        
        return self.a2, self.a1

        """
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = None # signals into hidden layer
        hidden_outputs = None # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = None # signals into final output layer
        final_outputs = None # signals from final output layer
        
        return final_outputs, hidden_outputs
        """

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###
        x = 1*X if len(X.shape) > 1 else X[None, :]
        derr = (y - self.a2)
        dh2 = derr # Scalar
        #print("dh2", dh2.shape)
        da1 = np.dot(dh2, self.W2.T)
        #print("da1", da1.shape)
        dW2 = np.dot(self.a1.T, dh2)
        #print("dW2", dW2.shape)
        dh1 = da1 * sigmoid_prime(self.h1)
        #print("dh1", dh1.shape)
        dW1 = np.dot(x.T, dh1)
        #print("dW1", dW1.shape)

        #print(delta_weights_h_o.shape, dW2.shape)
        delta_weights_i_h += dW1
        delta_weights_h_o += dW2
        
        """
        # TODO: Output error - Replace this value with your calculations.
        error = None # Output layer error is the difference between desired target and actual output.
        
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = None
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = None
        
        hidden_error_term = None
        
        # Weight step (input to hidden)
        delta_weights_i_h += None
        # Weight step (hidden to output)
        delta_weights_h_o += None
        """
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        #print("run")
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        self.forward_pass_train(features)
        if isinstance(self.a2, pd.DataFrame): # ugh
            return self.a2.reset_index()[0]
        else:
            return self.a2
    
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = None # signals into hidden layer
        hidden_outputs = None # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = None # signals into final output layer
        final_outputs = None # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 2000
learning_rate = 1e-0
hidden_nodes = 10
output_nodes = 1
