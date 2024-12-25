import numpy as np 


inputs = [1,2,3,2.5]
w1 = [0.2, 0.8, -0.5, 1]
w2 = [0.5,-0.91,0.26,-0.5]
w3 = [-0.26,-0.27,0.17,0.87]
b1 = 2
b2 = 3
b3 = 0.5

output = [
    #neuron 1
    inputs[0]*w1[0] + inputs[1]*w1[1]+ inputs[2]*w1[2] + inputs[3]*w1[3] + b1 ,
    #neuron 2
    inputs[0]*w2[0] + inputs[1]*w2[1]+ inputs[2]*w2[2] + inputs[3]*w2[3] + b2 ,
    #neuron3
    inputs[0]*w3[0] + inputs[1]*w3[1]+ inputs[2]*w3[2] + inputs[3]*w3[3] + b3 ,


]

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
[0.5, -0.91, 0.26, -0.5],
[-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

#print(np.dot(weights,inputs) + biases)
out = []
for w,b in zip(weights,biases):
    prod = np.dot(w,inputs) + b
    out.append(prod)
#print(out)       


inputs = [[1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
         [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

output = np.dot(inputs,np.array(weights).T) + biases
print(output)