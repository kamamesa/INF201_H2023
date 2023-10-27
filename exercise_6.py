# 
# 
# 

'''
Project task "Neural Network"
'''

# ---- SETUP ----

# Import
import numpy as np

# For the stolen code
from torchvision import datasets, transforms

# Vector of dimensions for each layer
n_vector = [784,512,256,10]

# Files to read
weights = ["W_"+str(number)+".txt" for number in range(1,len(n_vector))]
biases = ["b_"+str(number)+".txt" for number in range(1,len(n_vector))]

# ---- FUNCTIONS ----

# Reads text files into lists of values
def freader(wfile,bfile):
    with open(wfile) as W_f:
        W_raw = W_f.read()
    with open(bfile) as b_f:
        b_raw = b_f.read() 
    return W_raw.split(),b_raw.split(),W_raw.split("\n")[:-1]

# Sigma fn, today it does ramp.
def sigma(arg):
    return np.maximum(0,arg)

# ---- CLASSES ----

# Define layer: weight matrix, bias vector
class layer():
    def __init__(self,n,m):
        self.weight = np.ones((n,m))
        self.bias = np.ones(n)

    # Fill in data from specified files
    def read_file(self,w_f,b_f):
        u = freader(w_f,b_f)
        self.weight = u[0]
        self.bias = u[1]

    # Run evaluate on self with given input
    def evaluate(self,y):
        return sigma(self.weight @ y + self.bias)


# Create appropriate amount of layers and fill them in with the information given.
class network():        
    def __init__(self,dim):

        # Outbox
        self.layers = []

        # Create empty layers of given dimensions
        for i in range(len(dim)-1):
            the_layer = layer(dim[i],dim[i+1])

            # Put layer in layer list (aka "network")
            self.layers.append(the_layer)
    
    # Runs each layer's read-weight-from-file method. Feed it lists.
    def fill(self,w,b):
        for i in range(len(w)):
            self.layers[i].read_file(w[i],b[i])

    # Runs each layer's evaluate function
    def run(self):
        for i in range(len(self.layers)):
            self.layers[i].evaluate()


# ---- BODY ----
the_network = network(n_vector)
the_network.fill(weights,biases)


# Block nabbed from read.py
def get_mnist():
    return datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
def return_image(image_index, mnist_dataset):
    image, label = mnist_dataset[image_index]
    image_matrix = image[0].detach().numpy()
    return image_matrix.reshape(image_matrix.size), image_matrix, label
image_index = 19961
x, image, label = return_image(image_index, get_mnist())

