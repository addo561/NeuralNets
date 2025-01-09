import nnfs
import numpy as np   
from nnfs.datasets import spiral_data    
from Backpropagation import DenseLayer,softmax,activationRelu,loss,categorical_CrossEntropyLoss,Activation_Softmax_Loss_CategoricalCrossEntopy



# SGD optimizer
class Optimizer_SGD:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer

    def __init__(self, learning_rate=1., decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
                
    # Update parameters
    def update_params(self, layer):
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.biases += -self.current_learning_rate * layer.dbiases
        
    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
    