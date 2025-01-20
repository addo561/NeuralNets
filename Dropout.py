from L1_and_L2_regularization import *
import numpy as np   
from nnfs.datasets import spiral_data 

class Layer_Dropout:
    def __init__(self,rate):
        self.rate = 1 - rate
        
    def forward(self,inputs,training):
        self.inputs = inputs
        
        #training 
        if not training:
            self.output = inputs.copy()
            return 
        
        #scaled_mask
        self.binary_mask = np.random.binomial(1,self.rate,size=inputs.shape) / self.rate
        
        self.output = inputs * self.binary_mask
        
    def backward(self,dvlaues):
        #Gradient
        self.dinputs = dvlaues * self.binary_mask        


