from Backpropagation import *
from Adagrad import *
from RMSProp import * 
from Adam import * 
from SGD_momentum import *
from nnfs.datasets import spiral_data
import numpy as np 


class Dense_Layer:
    def __init__(self,n_inputs
                 ,n_neurons,
                 weight_regularizer_l1 = 0,
                 weight_regularizer_l2=0,
                 bias_regularizer_l1=0,
                 bias_regularizer_l2=0):
                #inintialize weight and bias
                self.weights = 0.01 * np.random.rand(n_inputs,n_neurons) # weights have shape(inputs,neurons)
                self.biases = np.zeros((1,n_neurons))# biases of row vector (1,n_neurons)

                #set regularization strength
                self.weight_regularizer_l1 = weight_regularizer_l1
                self.weight_regularizer_l2 = weight_regularizer_l2
                self.bias_regularizer_l1 = bias_regularizer_l1
                self.bias_regularizer_l2 = bias_regularizer_l2
    #forward pass 
    def ForwardPass(self,inputs):
            self.inputs = inputs
            self.output =  np.dot(inputs,self.weights) + self.biases #inputs * weights + biases giving you the neurons
    
    #backward pass
    def backward(self,dvalues):
        #gradients
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True)
        
        #Gradients on regularization 
        #l1 weight
        if self.weight_regularizer_l1 > 0:
            dl1 = np.ones_like(self.weight)
            dl1[self.weights < 0 ] = -1
            self.dweights += self.weight_regularizer_l1 * dl1
        
        #l2   weight
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights 
        
        #l1 on biases
        if self.biases_regularizer_l1 > 0 :
            dl1 = np.ones_like(self.biases)
            dl1[self.biases < 0 ] = -1
            self.dbiases += self.bias_regularizer_l1 * dl1
            
        #l2 bias
        if self.bias_regularizer_l2 > 0 : 
            self.biases += 2 * self.bias_regularizer_l2 * self.biases
        
        #gradients on values
        self.dinputs = np.dot(dvalues,self.weights.T)

        
    
    
    
    
class loss_E:
    def regularization_loss(self,layer):
        
        regularization_loss = 0
        
        # L1 regularization - weights
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
           
        # L2 regularization - weights
        if layer.weight_regularizer_l2 > 0: 
            regularization_loss += layer.weight_regularizer_l2 * np.sum((layer.weights**2))
            
        #l1 bias
        if layer.bias_regularizer_l1 > 0 :
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))        

        #l2 bias
        if layer.bias_regularizer_l2 > 0 :
            regularization_loss += layer.bias_regularizer_l2 * np.sum((layer.biases**2))
            
        return regularization_loss
    
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss  
    
    
DenseLayer = Dense_Layer
loss = loss_E  

X, y = spiral_data(samples=100, classes=3)
dense1 = DenseLayer(2, 64, weight_regularizer_l2=5e-4,
bias_regularizer_l2=5e-4)

activation1 = activationRelu()

dense2 = DenseLayer(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntopy()
optimizer = Optimizer_Adam(learning_rate=0.02, decay=5e-7)
if __name__ =='__main__':
    
    for epoch in range(1000):
        
        dense1.ForwardPass(X)
        activation1.forward(dense1.output)
        dense2.ForwardPass(activation1.output)
        data_loss = loss_activation.forward(dense2.output,y)
        
        regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
        
        loss = data_loss + regularization_loss
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions==y)
        if not epoch % 100:
                print(f'epoch: {epoch}, ' +
                f'acc: {accuracy:.3f}, ' +
                f'loss: {loss:.3f} (' +
                f'data_loss: {data_loss:.3f}, ' +
                f'reg_loss: {regularization_loss:.3f}), ' +
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

        