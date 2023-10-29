# 
# 
# 

'''
Project task "Neural Network"
'''

# ---- SETUP ----

# Import
import numpy as np

# Required for read.py fragment
from torchvision import datasets, transforms

# Vector of dimensions for each layer
n_vector = [784,512,256,10]

# List of files to read
weights = ["W_"+str(number)+".txt" for number in range(1,len(n_vector))]
biases = ["b_"+str(number)+".txt" for number in range(1,len(n_vector))]

# ---- STANDALONE FUNCTIONS ----

# Reads text files into lists of values
def freader(wfile,bfile):
    with open(wfile) as W_f:
        W_raw = W_f.read()
    with open(bfile) as b_f:
        b_raw = b_f.read() 
    return W_raw.split(),b_raw.split(),W_raw#.split("\n")[:-1]

# Sigma fn, today it does ramp.
def sigma(arg):
    return np.maximum(0,arg)

# ---- CLASSES ----

# Define layer: weight matrix, bias vector
class layer():
    def __init__(self,n,m):
        self.weight = np.ones((n,m))
        self.bias = np.ones(n)

    # Fill in data from specified weight/bias files.
    def read(self,w_f,b_f):
        u = freader(w_f,b_f)
        
        # CULPRIT
        self.weight = 

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
    
    # Runs each layer's read-weight-from-file method. Feed it lists of lists.
    def read(self,w,b):
        for i in range(len(w)):
            self.layers[i].read(w[i],b[i])

    # Runs each layer's evaluate function
    def evaluate(self,initial_y):
        print(f"weight len: {len(self.layers[0].weight)}")   # debug
        print(f"initial y len: {len(initial_y)}")               # debug
        self.layers[0].evaluate(initial_y)

        for i in range(1,len(self.layers)-1):
            self.layers[i].evaluate(self.layers[i-1])


# ---- MNIST BLOCK FROM READ.PY ----

def get_mnist():
    return datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
def return_image(image_index, mnist_dataset):
    image, label = mnist_dataset[image_index]
    image_matrix = image[0].detach().numpy()
    return image_matrix.reshape(image_matrix.size), image_matrix, label
image_index = 19961
x, image, label = return_image(image_index, get_mnist())


# ---- BODY ----

# Create and fill a network from the files
the_network = network(n_vector)
the_network.read(weights,biases)

#x = x.tolist()
#print(len(the_network.layers[0].weight))
the_network.evaluate(x)

# Do the input dance
#the_network.evaluate(x)






