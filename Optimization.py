import numpy as np 
from nnfs.datasets import spiral_data,vertical_data 
import math
from AddingLayers import DenseLayer
from ActivationFunctions import softmax,activationRelu
from Loss import loss,categorical_CrossEntropyLoss


#log loss 
softmax_activation = [0.7,0.2,0.4]
target = [1,0,0]
loss = -(math.log(softmax_activation[0]))#since it is the correct class
#print(loss)



            
X,y = vertical_data(samples=100,classes=3)
#plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
#plt.show()
    
dense1 = DenseLayer(2,3)
acc1 = activationRelu()
dense2 = DenseLayer(3,3)
acc2 = softmax()

#create loss function
loss_function = categorical_CrossEntropyLoss()


#Optimize
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()


for i in range(1000):
    #update weights
    dense1.weights +=  0.05 * np.random.randn(2,3)
    dense1.biases += 0.05 * np.random.randn(1,3)
    dense2.weights +=  0.05 * np.random.randn(3,3)
    dense2.biases += 0.05 * np.random.randn(1,3)
    
    dense1.ForwardPass(X)
    acc1.forward(dense1.output)
    dense2.ForwardPass(acc1.output)
    acc2.forward(dense2.output)
    
    loss = loss_function.calculate(acc2.output,y)
    predictions = np.argmax(acc2.output,axis=1)
    accuracy = np.mean(predictions==y)
    
    accuracy_stop = 0.94
    if math.isclose(accuracy,accuracy_stop):
        break
    if loss<lowest_loss:
        print(f'iteration: {i} loss: {loss} accuracy:{accuracy}')
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss= loss
    #revert weights and biases
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
                
