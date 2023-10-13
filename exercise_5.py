# 20231013, utf-8, Spyder
# INF201 Exercise 5

__author__ = "fill in locally"
__email__ = "fill in locally"

'''
Task 1
'''

# Import
import numpy as np

# This is our sigma function, in this case just ReLU/ramp.
# np.maximum returns 0 if arg<0, or arg itself if it is not.
def sigma(arg):
    return np.maximum(0,arg)

'''
Non-general method (pts 1 & 2):
'''

def layer_nongen(x, W, b):
    return sigma(W @ x + b)

def NN(x, W_1, W_2, b_1, b_2):
    y_1 = layer_nongen(x, W_1, b_1)
    y_2 = layer_nongen(y_1, W_2, b_2)
    return y_2

# Values for NN with one and two layers
x = np.random.rand(64)
W_1 = np.random.rand(10, 64)
b_1 = np.random.rand(10)
W_2 = np.random.rand(10, 10)
b_2 = np.random.rand(10)

print("y_2 with manual method:")
print(NN(x, W_1, W_2, b_1, b_2),"\n")

'''
General method for y_L (pts 3 & 4):
'''

# "n" the list of matrix sizes, these are the layers of the network.
n_vector = [64, 128, 128, 128, 10]
# n amount of layers in the vector. In this case there are 5 layers
L = len(n_vector)

# Containers
Ws = []
ys = []

# "Layer" function runs sigma on previous weight, bias, and the previous layer.
#     [-1] retrieves the most recent entry, as we assume we're working on the last entry.
#     [i-1] would mean the same, but add i needlessly since we're building the matrices as we go.
# We could also write this as np.add(np.dot(Ws[-1],ys[-1]),bs[-1]), but that's longer and uglier.

def layer(Ws, ys, bs):
    return sigma(Ws[-1] @ ys[-1] + bs[-1])

# First input vector. Random contents.
ys.append(np.random.rand(n_vector[0], 1))

# Cycling through all the matrix sizes we want, starting at 1 because y_0 is already in ys.
for i in range(1, L):
    # Generate specific size of matrix, as specified in n_vector.
    Ws.append(np.random.rand(n_vector[i], n_vector[i-1]))
    b = np.random.rand(n_vector[i], 1)

    # Label and print the dimensions of the current W. Unwrap tuple w/ * for aesthetics.
    print("Weight matrix W_" + str(len(Ws)) + ":", *Ws[-1].shape, "\n")

    # Stores new layer based on previous layer.
    ys.append(layer(Ws,ys,b))

    # Pseudo-destructor eats oldest Ws and ys, which are no longer needed.
    # We don't really need to store the previous W at all, but doing so lets us easily get its shape.
    # We could also just let the lists grow, but that's bad practice.
    if i != 1:
        del Ws[0]
        del ys[0]

# The most recent y is y_L. Let's print it.
y_L = ys[-1]
print("Final layer y_L:")
print(y_L)
