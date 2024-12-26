import nnfs
import numpy as np
from nnfs.datasets import spiral_data #nnfs library fro plotting
import matplotlib.pyplot as plt
nnfs.init()

X,y = spiral_data(samples=100,classes=3)
plt.scatter(X[:,0],X[:,1])
plt.show()




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

Layer1 = np.dot(inputs,np.array(weights).T) + biases
Layer2 = np.dot(Layer1,np.array(weights2).T) + biases2
print(Layer2)


