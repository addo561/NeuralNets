import numpy as np
import nnfs
from nnfs.datasets import spiral_data #nnfs library fro plotting
import matplotlib.pyplot as plt
nnfs.init()



inputs = [[1.0, 2.0, 3.0, 2.5],[2.0, 5.0, -1.0, 2.0],[-1.5, 2.7, 3.3, -0.8]]

#creating 2 hidden layers
weights = [[0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]]

weights2 = [[0.4, 0.8, -0.6],
        [0.5, -0.9, 0.26],
        [-0.5, -0.27, 0.17]]
biases = [2.0, 3.0, 0.5]
biases2 = [1.0,5.5,0.7]

Layer1 = np.dot(np.array(inputs),np.array(weights).T) + biases
Layer2 = np.dot(Layer1,np.array(weights2).T) + biases2
#print(Layer2)






#class of layer
class DenseLayer:
        def __init__(self,n_inputs,n_neurons):
                self.weights = 0.01 * np.random.rand(n_inputs,n_neurons) # weights have shape(inputs,output)
                self.biases = np.zeros((1,n_neurons))# biases of row vector (1,n_neurons)

        def ForwardPass(self,inputs):
                self.output =  np.dot(inputs,self.weights) + biases #inputs * weights + biases giving you the neurons
        

X,y = spiral_data(samples=100,classes=3)
#plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
#plt.show()

dense1 = DenseLayer(2,3)
dense1.ForwardPass(X)# X from the spiral Data#
print(dense1.output[:5])