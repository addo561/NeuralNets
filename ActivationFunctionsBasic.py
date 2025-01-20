import numpy as np  
import nnfs 
from nnfs.datasets import spiral_data
from AddingLayers import DenseLayer #import Dense layer


#Relu class -> max(0,x)
class activationRelu:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)
    
    
#softmax
class softmax:
    def forward(self,inputs):
        #unnoramlized probability
        exp_vals = np.exp(inputs-np.max(inputs,axis=1,keepdims=True)) #-np.max to prevent exploding  values
        #normalize
        self.output = exp_vals/np.sum(exp_vals,axis=1,keepdims=True)    
    
X,y = spiral_data(samples=100,classes=3)
#plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
#plt.show()
