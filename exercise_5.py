# 
# 
# 

import numpy as np

# Edit this to change what happens to each layer

def sigma (arg):
    return 0.0 if arg<=0 else arg
vec_sigma = np.vectorize(sigma)

# gives y
def layer (x, W, b):
    W.reshape(W.shape[0],W.shape[1])
    return vec_sigma(W @ x + b)

'''
# works 2 layers
def fn (x, W_1, W_2, b_1, b_2):
    y_1 = layer(x, W_1, b_1)
    y_2 = layer(y_1, W_2, b_2)
    return y_2

# Setup initial
x = np.random.rand(64)
W_1 = np.random.rand(10,64)
b_1 = np.random.rand(10)
'''

n = [64, 128, 128, 128, 10]

x = np.random.rand(n[0])
W = np.random.rand(n[1],n[0])
b = np.random.rand(n[1])
first = layer(x,W,b)


def NN (FLAG,W,b,weights,prev=first):

    if FLAG:
        prev = np.random.rand(n[0])
    else:
        for i in n[:]:
            W = np.random.rand(i,W.shape[0])
            b = np.random.rand(i)
            layer(prev,W,b)
        
            NN(0,W,b,weights,prev)

    return prev

print(NN(1,W,b,n))

