import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Dropout import *
from Adam import Optimizer_Adam
from L1_and_L2_regularization import *
from Sigmoid import *

nnfs.init()

# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons, 
                 weight_regularizer_l1=0, weight_regularizer_l2=0, 
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs,training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Layer_input:
    def forward(self,inputs,training):
        self.output  = inputs  
        
        
# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs,training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable, 
        # let's make a copy of values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self,outputs):
        return outputs
    
    
# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs,training):
        # Remember input values
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    def predictions(self,outputs):
        return np.argmax(outputs,axis=1)



# Common loss class
class Loss:
    # Regularization loss calculation
    def regularization_loss(self):
        # 0 by default
        regularization_loss = 0
        
        # Calculate regularization loss
        # iterate all trainable layers
        for layer in self.trainable_layers:
        # L1 regularization - weights
        # calculate only when factor greater than 0
        
        
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                                    np.sum(np.abs(layer.weights))
            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                    np.sum(layer.weights * layer.weights)
            # L1 regularization - biases
            # calculate only when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                    np.sum(np.abs(layer.biases))
            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                                    np.sum(layer.biases * layer.biases)
        return regularization_loss
    
    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers


    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y,*, include_regularization=False):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        
        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return loss
        return data_loss,self.regularization_loss()


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values - 
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Softmax classifier - combined Softmax activation 
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():


    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Common accuracy class
class Accuracy:
# Calculates an accuracy
# given predictions and ground truth values
    def calculate(self, predictions, y):
    # Get comparison results
        comparisons = self.compare(predictions, y)
    # Calculate an accuracy
        accuracy = np.mean(comparisons)
# Return accuracy
        return accuracy



# Accuracy calculation for classification model
class Accuracy_Categorical(Accuracy):
    # No initialization is needed
    def init(self, y):
        pass
        # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y



# Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):
    def __init__(self):
    # Create precision property
        self.precision = None
    # Calculates precision value
    # based on passed in ground truth values
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    
    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision




# Model class
class Model:
    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None
    
    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)
    
    # Set loss, optimizer, and accuracy
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    # Finalize the model
    def finalize(self):
        # Create and set the input layer
        self.input_layer = Layer_input()
        # Count all the objects
        layer_count = len(self.layers)
        # Initialize a list containing trainable layers
        self.trainable_layers = []
        
        # Iterate over the objects
        for i in range(layer_count):
            # If it's the first layer, the previous layer is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            # All layers except for the first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            # The last layer
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            
            # If layer contains an attribute called "weights", it's a trainable layer
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        
        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)
        
        # If output activation is Softmax and loss function is Categorical Cross-Entropy,
        # create an object of combined activation and loss function
        if isinstance(self.layers[-1], Activation_Softmax) and \
           isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    # Train the model
    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        # Initialize accuracy object
        self.accuracy.init(y)
        
        # Main training loop
        for epoch in range(1, epochs + 1):
            # Perform the forward pass
            output = self.forward(X, training=True)
            
            # Calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss
            
            # Get predictions and calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)
            
            # Perform backward pass
            self.backward(output, y)
            
            # Optimize (update parameters)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()
            
            # Print a summary
            if epoch % print_every == 0:
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate}')
            
        # If there is validation data
        if validation_data is not None:
            # For better readability
            X_val, y_val = validation_data
            # Perform the forward pass
            output = self.forward(X_val, training=False)
            # Calculate the loss
            loss = self.loss.calculate(output, y_val)
            # Get predictions and calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)
            # Print a summary
            print(f'validation, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f}')

    # Perform forward pass
    def forward(self, X, training):
        # Call forward method on the input layer
        self.input_layer.forward(X, training)
        
        # Call forward method of every object in a chain
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        # Return output of the last layer
        return layer.output

    # Perform backward pass
    def backward(self, output, y):
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method on the combined activation/loss
            self.softmax_classifier_output.backward(output, y)
            # Set dinputs in the last layer
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
        
        # Call backward method going through all the objects but last
        for layer in reversed(self.layers[:-1]):
            layer.backward(layer.next.dinputs)
        
        # Call backward method on the loss
        self.loss.backward(output, y)
        
        # Call backward method going through all the objects in reversed order
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


# Create dataset
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())

# Set loss, optimizer, and accuracy objects
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)
