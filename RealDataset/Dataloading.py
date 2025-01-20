import os  
import cv2 
import matplotlib.pyplot as plt 
import numpy as np  
  
labels = os.listdir('fashion_mnist_images/train')
#print(labels)
files = os.listdir('fashion_mnist_images/train/0')
#print(files[:10])
#print(len(files))


def load_mnist_dataset(dataset,path):
    
    labels = os.listdir(os.path.join(path,dataset))
    
    # Create lists for samples and labels
    X = []
    y = []
    
     #For each label folder
    for label in labels:
    # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
        # Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file),
                cv2.IMREAD_UNCHANGED)
            # And append it and a label to the lists
            X.append(image)
            y.append(label)
    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

# MNIST dataset (train + test)
def create_data_mnist(path):
    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    # And return all the data
    return X, y, X_test, y_test    



