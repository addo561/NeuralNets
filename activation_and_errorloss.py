import numpy as np         
from nnfs.datasets import sine_data    
from L1_and_L2_regularization import *
from Sigmoid import  *
from BinaryCrossEntropyloss import *

class Activation_linear:
    def forward(self,inputs,training):
        self.inputs= inputs
        self.output= inputs
        
    def backward(self,dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule

        self.dinputs = dvalues.copy()
    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs    

class mean_squared_error(Loss):
    def forward(self,y_pred,y_true):
        sample_losses = np.mean((y_true- y_pred)**2,axis=-1)
        return sample_losses
    
    def backward(self,dvalues,y_true):
        #sample number 
        samples=len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them

        outputs = len(dvalues[0])
        #Gradient
        self.dinputs = -2 * ( y_true - dvalues)/outputs      
        #normalize grad
        
        self.dinputs = self.dinputs/samples    
        
class mean_abs_error(Loss):
    def forward(self,y_pred,y_true):
        sample_losses = np.mean(np.abs(y_true- y_pred)**2,axis=-1)
        return sample_losses        
    
    def backward(self,dvalues,y_true):
        
        samples = len(dvalues)
        
        outputs = len(dvalues[0])
        
        
        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
  
  
  
  
        
X, y = sine_data()
dense1 = Layer_Dense(1, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 64)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(64, 1)
activation3 = Activation_linear()

loss_function = mean_squared_error()
optimizer = Optimizer_Adam()
        
#accuracy prediction
accuracy_precision = np.std(y)/250

for epoch in range(10001):
    

    
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    data_loss = loss_function.calculate(activation3.output, y)
    regularization_loss = \
    loss_function.regularization_loss(dense1) + \
    loss_function.regularization_loss(dense2) + loss_function.regularization_loss(dense3)

    loss = data_loss + regularization_loss
    
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) <
    accuracy_precision)  
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f} (' +
            f'data_loss: {data_loss:.3f}, ' +
            f'reg_loss: {regularization_loss:.3f}), ' +
            f'lr: {optimizer.current_learning_rate}')
        
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
        

    
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

#try on test data
import matplotlib.pyplot as plt
X_test, y_test = sine_data()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)
plt.plot(X_test, y_test)
plt.plot(X_test, activation3.output)
plt.show()