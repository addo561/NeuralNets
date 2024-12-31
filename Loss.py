import numpy as np 
from nnfs.datasets import spiral_data 
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


            
X,y = spiral_data(samples=100,classes=3)
#plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
#plt.show()
    
dense1 = DenseLayer(2,3)
acc1 = activationRelu()
dense2 = DenseLayer(3,3)
acc2 = softmax()
loss_function = categorical_CrossEntropyLoss()

#layers
dense1.ForwardPass(X)# X from the spiral Data#
acc1.forward(dense1.output)#pass output to relu
dense2.ForwardPass(acc1.output)#pass output of relu to second layer
acc2.forward(dense2.output)#apply softmax on second layer output

print(acc2.output[:5])
loss = loss_function.calculate(acc2.output,y)
print(f'loss:',{loss})