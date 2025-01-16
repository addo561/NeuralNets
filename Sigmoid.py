import numpy as np         
from nnfs.datasets import spiral_data    
from L1_and_L2_regularization import *


class Activation_sigmoid:
    
    def forward(self,inputs):
        self.inputs = inputs 
        self.output = 1/(1+np.exp(-inputs))
        
    def backward(self,dvalues):
        self.dinputs = dvalues * (1-self.output) * self.output    