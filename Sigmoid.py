import numpy as np         
from nnfs.datasets import spiral_data    
from L1_and_L2_regularization import *


class Activation_sigmoid:
    
    def forward(self,inputs,training):
        self.inputs = inputs 
        self.output = 1/(1+np.exp(-inputs))
        
    def backward(self,dvalues):
        self.dinputs = dvalues * (1-self.output) * self.output    
        
    def predictions(self,outputs):
        return (outputs >0.5) * 1
    
        