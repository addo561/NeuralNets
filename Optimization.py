import numpy as np 
from nnfs.datasets import spiral_data,vertical_data 
import math


#log loss 
softmax_activation = [0.7,0.2,0.4]
target = [1,0,0]
loss = -(math.log(softmax_activation[0]))#since it is the correct class
#print(loss)


# fitering correct labels,finding confidences and mean
#class of layer
class DenseLayer:
        def __init__(self,n_inputs,n_neurons):
                self.weights = 0.01 * np.random.rand(n_inputs,n_neurons) # weights have shape(inputs,neurons)
                self.biases = np.zeros((1,n_neurons))# biases of row vector (1,n_neurons)

        def ForwardPass(self,inputs):
                self.output =  np.dot(inputs,self.weights) + self.biases #inputs * weights + biases giving you the neurons
        


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

#Common loss
class loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

#cross-entropyloss
class categorical_CrossEntropyLoss(loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        clipped_ypred= np.clip(y_pred,1e-7,1-1e-7)
        
        if len(y_true.shape)==1:
            correct_confidence = clipped_ypred[range(samples),y_true]
        elif len(y_true.shape)==2:
            correct_confidence= np.sum(clipped_ypred * y_true ,axis=1)    
        
        neg_log_likelihoods = -np.log(correct_confidence)
        return neg_log_likelihoods


            
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
                
