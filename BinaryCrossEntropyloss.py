import numpy as np         
from nnfs.datasets import spiral_data    
from L1_and_L2_regularization import *
from Sigmoid import  *

class Loss_BinaryCrossentropy(Loss):
    
    def forward(self,y_pred, y_true):
        
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                        (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses,axis=-1)
        
        return sample_losses
    
    def backward(self,dvalues,y_true):
        
        samples = len(dvalues)
        
        outputs = len(dvalues[0])
        
        clipped_dvalues =  np.clip(dvalues,1e-7,1-1e-7)
        
        
        self.dinputs = -(y_true / clipped_dvalues -
                        (1 - y_true) / (1 - clipped_dvalues)) / outputs
        
        self.dinputs = self.dinputs / samples
        
X, y = spiral_data(samples=100, classes=2)

y = y.reshape(-1,1)

dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4,
bias_regularizer_l2=5e-4)


activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 1)

activation2 = Activation_sigmoid()

loss_function = Loss_BinaryCrossentropy()

optimizer = Optimizer_Adam(decay=5e-7)

for epoch in range(10001):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    data_loss = loss_function.calculate(activation2.output, y)
    regularization_loss = \
    loss_function.regularization_loss(dense1) + \
    loss_function.regularization_loss(dense2)


    loss = data_loss + regularization_loss
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions==y)

    
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, '+
        f'loss: {loss:.3f} (' +
        f'data_loss: {data_loss:.3f}, ' +
        f'reg_loss: {regularization_loss:.3f}), ' +
        f'lr: {optimizer.current_learning_rate}')

    
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
