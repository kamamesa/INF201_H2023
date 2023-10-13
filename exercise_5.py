# 20231011, utf-8, Spyder
# INF201 Exercise 5

__author__ = "Marcus D. Figenschou, Katla M. Meyer"
__email__ = "marcus.dalaker.figenschou@nmbu.no, katla.maria.meyer@nmbu.no"


# Import
import numpy as np

'''
Task 1 and 2
'''

# values for NN with one and two layers
x = np.random.rand(64)
W_1 = np.random.rand(10, 64)
b_1 = np.random.rand(10)
W_2 = np.random.rand(10, 10)
b_2 = np.random.rand(10)


# This is the relu function.
def relu(s):
    # using the np.maximum. This takes the maximum of 0 and s, element wise in s. If its greater than 0 it will return s, 
    # if it's smaller than 0 it will return 0.
    return np.maximum(0,s)

def layer_task1(x, W, b):
    return relu(W @ x + b)

def NN(x, W_1, W_2, b_1, b_2):
    y_1 = layer_task1(x, W_1, b_1)
    y_2 = layer_task1(y_1, W_2, b_2)
    return y_2

print("2 layered output:")
print(NN(x, W_1, W_2, b_1, b_2))
print("\t")


'''
Task 3-4
'''

# "n" the list of matrix sizes, these are the layers of the network.
n_vector = [64, 128, 128, 128, 10]
# n amount of layers in the vector. In this case there are 5 layers
L = len(n_vector)

# Containers
Ws = []
bs = []
ys = []


# First input vector. Random contents.
ys.append(np.random.rand(n_vector[0], 1))

# Function that runs relu on previous weight, bias, and the previous layer.
#     [-1] retrieves the most recent entry, as we assume we're working on the last entry.
#     [i-1] would mean the same, but add i needlessly since we're building the matrices as we go.
# We could also write this as np.add(np.dot(Ws[-1],ys[-1]),bs[-1]), but that's longer and uglier.

def layer(Ws, ys, bs):
    return relu(Ws[-1] @ ys[-1] + bs[-1])


# Cycling through all the matrix sizes we want, starting at 1 because y_0 is already in ys.
for i in range(1, L):
    # Generate specific size of matrix, as specified in n_vector.
    Ws.append(np.random.rand(n_vector[i], n_vector[i - 1]))
    bs.append(np.random.rand(n_vector[i], 1))

    # Label and print the dimensions of the current W. Unwrap tuple w/ * for aesthetics.
    print("Weight matrix W_" + str(len(Ws)) + ":", *Ws[-1].shape, "\n")

    # Make new layer based on previous layer. layer() can find the newest of the layers itself.
    Y_now = layer(Ws, ys, bs)

    # Put result
    ys.append(Y_now)

# The most recent y is y_L, let's print it.
y_L = ys[-1]
print("Final layer y_L:")
print(y_L)
