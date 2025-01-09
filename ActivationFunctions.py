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
    
dense1 = DenseLayer(2,3)
acc1 = activationRelu()
dense2 = DenseLayer(3,3)
acc2 = softmax()

dense1.ForwardPass(X)# X from the spiral Data#
acc1.forward(dense1.output)#pass output to relu

print('acc1 :',acc1.output[:5])

dense2.ForwardPass(acc1.output)#pass output of relu to second layer
acc2.forward(dense2.output)#apply softmax on second layer output

print('acc2 :',acc2.output[:5])