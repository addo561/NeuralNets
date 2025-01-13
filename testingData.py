from Backpropagation import DenseLayer,softmax,activationRelu,loss,categorical_CrossEntropyLoss,Activation_Softmax_Loss_CategoricalCrossEntopy
from nnfs.datasets import spiral_data
import numpy as np

X_test,y_test = spiral_data(classes=3,samples=100)
# Create Dense layer with 2 input features and 64 output values
dense1 = DenseLayer(2, 64)

# Create ReLU activation (to be used with Dense layer):
activation1 = activationRelu()

# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = DenseLayer(64, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntopy()
# Perform a forward pass of our testing data through this layer

dense1.ForwardPass(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here

activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs

dense2.ForwardPass(activation1.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss

loss = loss_activation.forward(dense2.output, y_test)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis 

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')