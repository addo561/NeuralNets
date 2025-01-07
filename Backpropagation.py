import numpy as np   
import nnfs 
from nnfs.datasets import spiral_data

# fitering correct labels,finding confidences and mean
#class of layer
class DenseLayer:
        def __init__(self,n_inputs,n_neurons):
                self.weights = 0.01 * np.random.rand(n_inputs,n_neurons) # weights have shape(inputs,neurons)
                self.biases = np.zeros((1,n_neurons))# biases of row vector (1,n_neurons)

        #forward pass 
        def ForwardPass(self,inputs):
                self.inputs = inputs
                self.output =  np.dot(inputs,self.weights) + self.biases #inputs * weights + biases giving you the neurons
        
        #backward pass
        def backward(self,dvalues):
            #gradients
            self.dweights = np.dot(self.inputs.T,dvalues)
            self.dbiases = np.sum(dvalues,axis=0,keepdims=True)
            #gradients on values
            self.dinputs = np.dot(dvalues,self.weights.T)
            
#Relu class -> max(0,x)
class activationRelu:
    def forward(self,inputs):
        self.inputs  = inputs
        self.output = np.maximum(0,inputs)
    
    #backward
    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
        
        
#softmax
class softmax:
    def forward(self,inputs):
        self.inputs = inputs
        #unnoramlized probability
        exp_vals = np.exp(inputs-np.max(inputs,axis=1,keepdims=True)) #-np.max to prevent exploding  values
        #normalize
        self.output = exp_vals/np.sum(exp_vals,axis=1,keepdims=True)    

    
    #backward
    def backward(self,dvalues):
        #create uninitialized array
        self.dinputs = np.empty_like(self.inputs)
        for index,(single_output,single_dvalues) in enumerate(zip(self.output,dvalues)):
            single_output = single_output.reshape(-1,1)# Flatten output array
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output,single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)


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

    def backward(self,dvalues,y_true):
        #Number of samples
        samples = len(dvalues)
        
        #NUmber of in every sample
        labels = len(dvalues[0])
        
        # If labels are sparse, turn them into one-hot vector

        if len(y_true.shape)==1:
            y_true = np.eye(labels)[y_true]
        
        
        #Gradients
        self.dinputs = -y_true / dvalues
        #Normalize
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossEntopy():
    def __init__(self):
        self.activation = softmax()
        self.loss = categorical_CrossEntropyLoss()
        
    #forward
    def forward(self,inputs,y_true):
        #activation layer output
        self.activation.forward(inputs)
        self.output = self.activation.output
        #loss  output
        return self.loss.calculate(self.output,y_true)

    def backward(self,dvalues,y_true):
        #sample
        samples = len(dvalues)
        #one-hot-encode labels
        if len(y_true.shape)==2:
            y_true = np.argmax(y_true,axis=1)
        
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples
    



X,y = spiral_data(samples=100,classes=3)
#plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
#plt.show()
    
dense1 = DenseLayer(2,3)
acc1 = activationRelu()
dense2 = DenseLayer(3,3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntopy()


'''acc2 = softmax()
loss_function = categorical_CrossEntropyLoss()
'''
#layers
dense1.ForwardPass(X)# X from the spiral Data#
acc1.forward(dense1.output)#pass output to relu
dense2.ForwardPass(acc1.output)#pass output of relu to second layer
# acc2.forward(dense2.output) apply softmax on second layer output

loss = loss_activation.forward(dense2.output,y)

# Let's see output of the first few samples:
print(loss_activation.output[:5])

print(f'loss:',{loss})
    
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
 y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
print('acc: ',accuracy)    


#backward
loss_activation.backward(loss_activation.output,y)
dense2.backward(loss_activation.dinputs)
acc1.backward(dense2.dinputs)
dense1.backward(acc1.dinputs)


# Print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)
