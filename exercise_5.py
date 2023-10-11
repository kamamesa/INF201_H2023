# 20231011, utf-8, Spyder
# NAME AND EMAIL ADDRESS
# NAME AND EMAIL ADDRESS
# INF201 Exercise 5


'''
Task 1
'''

# Import
import numpy as np

# "n" the list of matrix sizes
n_vector = [64,128,128,128,10]
L = len(n_vector)

# Containers
Ws = []
ys = []
bs = []

# First input vector. Random contents.
ys.append ( np.random.rand(n_vector[0],1) )

# Function that does whatever. Today, it zeroes arg if it's zero or negative.
def sigma(arg):
    arg[arg<=0] = 0.0
    return arg

# Function that runs sigma() on previous weight, biase, and the previous layer.
#     [-1] retrieves the most recent entry, as we assume we're working on the last entry.
#     [i-1] would mean the same, but add i needlessly since we're building the matrices as we go.
# We could also write this as np.add(np.dot(Ws[-1],ys[-1]),bs[-1]), but that's longer and uglier.
def layer(Ws,ys,bs):
    return sigma(Ws[-1] @ ys[-1] + bs[-1])


# Cycling through all the matrix sizes we want, starting at 1 because y_0 is already in ys.
for i in range(1, L):

    # Generate specific size of gibberish matrix, as specified in n_vector.
    Ws.append(np.random.rand(n_vector[i], n_vector[i-1]))
    bs.append(np.random.rand(n_vector[i], 1))
    
    # Label and print the dimensions of the current W. Unwrap tuple w/ * for aesthetics.
    print("Weight matrix W_"+str(len(Ws))+ ":", *Ws[-1].shape, "\n")

    # Make new layer based on previous layer. layer() can find the newest of the layers itself.
    Y_now = layer(Ws,ys,bs)

    # Put result
    ys.append(Y_now)

# The most recent y is y_L, let's print it.
y_L = ys[-1]
print("Final layer y_L:")
print(y_L)
