#import all classes
from Backpropagation import DenseLayer,softmax,activationRelu,loss,categorical_CrossEntropyLoss,Activation_Softmax_Loss_CategoricalCrossEntopy
import numpy as np  
import nnfs 
from nnfs.datasets import spiral_data


class Optimizer_SGD:
    #initialize optimizer
    def __init__(self,learning_rate=1,decay=0.,momentum=0.):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.momentum = momentum
    
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            
    #update params
    def update_params(self,layer):
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer,'weight_momentums'):
                #momentums for weight and bias
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
            
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases 
            
        #update weights and biases
        layer.weights += weight_updates
        layer.biases += bias_updates             
        
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
        
        
        
        



# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 64 output values
dense1 = DenseLayer(2, 64)

# Create ReLU activation (to be used with Dense layer):
activation1 = activationRelu()

# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = DenseLayer(64, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntopy()

# Create optimizer
optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)

if __name__ =='__main__':
# Train in loop
    for epoch in range(10001):
        # Perform a forward pass of our training data through this layer
        dense1.ForwardPass(X)

        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forward(dense1.output)


        # Perform a forward pass through second Dense layer
        # takes outputs of activation function of first layer as inputs
        dense2.ForwardPass(activation1.output)
        # Perform a forward pass through the activation/loss function
        # takes the output of second dense layer here and returns loss
        loss = loss_activation.forward(dense2.output, y)
        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions==y)
        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
                f'acc: {accuracy:.3f}, ' +
                f'loss: {loss:.3f}, ' +
                f'lr: {optimizer.current_learning_rate}')
            
        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        
        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
            