# 
# 
# 

'''
Project task "Neural Network"
'''

# ---- SETUP ----

# Import
import numpy as np

# Vector of dimensions for each layer
n_vector = [784,512,256,10]

# Files to read
weights = ["W_"+str(number)+".txt" for number in range(1,3+1)]
biases = ["b_"+str(number)+".txt" for number in range(1,3+1)]

# ---- FUNCTIONS ----

# Reads text files into lists of values
def file_reader(wfile,bfile):
    with open(wfile) as W_f:
        W_raw = W_f.read()
    with open(bfile) as b_f:
        b_raw = b_f.read() 
    return W_raw.split(),b_raw.split(),W_raw.split("\n")[:-1]


# Sigma fn, today it does ramp.
def sigma(arg):
    return np.maximum(0,arg)


# FIX
# Apply function should take 2 layers and apply sigma to one based on another
def apply(prev,present):

    return sigma(layer,)

# FIX
# Unsure what "evaluating" is supposed to do
def evaluate(x):
    # what
    return 1

# ---- CLASSES ----

# Define layer, each has a weight matrix and a bias vector
class layer():
    def __init__(self,n,m,b):
        self.weight = np.zeros((n,m))
        self.bias = np.zeros(b)
        self.n = self.weight.shape[0]
        self.m = self.weight.shape[1]

# Create appropriate amount of layers and fill them in with the information given.
class network():        
    def __init__(self,dim,ws,bs):
            
        # Read weights and biases from files. Sample first row of w for row length.
        temp = file_reader(ws,bs)
        w = temp[0]; b = temp[1]; rowlength = len(temp[2][0])        
        
        # Outbox
        self.layers = []
        
        # Creating and filling in each layer:

        # Create empty matrix of given dimensions
        for i in range(len(dim)-1):
            the_layer = layer(dim[i],dim[i+1],dim[i+1])
    
            # Enter weights into the matrix
            for i in enumerate(w):
                if i[0] < the_layer.weight.shape[1]:
                    row = (i[0] // rowlength)       # What row we're on
                    ptr = i[0] - (row*rowlength)    # Position in row
                    the_layer.weight[row][ptr] = i[1]
            
            # Enter biases into the matrix
            for i in enumerate(b):
                if i[0] < the_layer.bias.shape[0]:
                    the_layer.bias[i[0]] = b[i[0]]

            # Put layer in pile of layers in the network            
            self.layers.append(the_layer)


#---------------------
