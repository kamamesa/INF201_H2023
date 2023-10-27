import numpy as np
from torchvision import datasets, transforms

def ReLu(arg):
    return np.maximum(0, arg)

class Layer:
    def __init__(self, n, m):
        self.W = np.zeros((n, m))
        self.b = np.zeros(n)

    def read(self, nameW, nameb):
        try:
            with open(nameW, "r") as weight_file:
                self.W = np.loadtxt(weight_file)
            with open(nameb, "r") as bias_file:
                self.b = np.loadtxt(bias_file)
            '''
            activate these lines for debugging:
            
            print("Read weights for layer:", nameW)
            print("Weight shape:", self.W.shape)
            '''
        except FileNotFoundError:
            print(f"Error: Could not find the weight or bias file ({nameW} or {nameb}).")

class Network:
    def __init__(self, n_vector):
        self.layers = [Layer(n_vector[i], n_vector[i + 1]) for i in range(len(n_vector) - 1)]

    def read(self, weights_prefix, biases_prefix, num_sets):
        for set_number in range(1, num_sets + 1):
            for i, layer in enumerate(self.layers):
                weight_filename = f"{weights_prefix}_{i + 1}.txt"
                bias_filename = f"{biases_prefix}_{i + 1}.txt"
                layer.read(weight_filename, bias_filename)

    def evaluate(self, x):
        for layer in self.layers:
            if x.shape[0] != layer.W.shape[1]:
                print("Input dimension does not match layer dimension.")
                return
            x = ReLu(layer.W @ x + layer.b)
        return x

# Define the dimensions for each layer
n_vector = [784, 512, 256, 10]

# Create a network with the specified layer dimensions
my_network = Network(n_vector)

# Specify the file prefixes for weights and biases
weights_prefix = "W"
biases_prefix = "b"

# Specify the number of weight and bias sets you have (e.g., 3 sets)
num_sets = 3

# Read the weights and biases for all layers for each set
my_network.read("/Users/marcusdalakerfigenschou/Documents/NMBU/3-året/INF201/Oppgaver/NN/NeuralNetProject/canvas/W",
                "/Users/marcusdalakerfigenschou/Documents/NMBU/3-året/INF201/Oppgaver/NN/NeuralNetProject/canvas/b", num_sets)

# Create an example input x (adjust as needed)
input_x = np.random.rand(784)  # Replace with your desired input

# Evaluate the neural network with the input
output = my_network.evaluate(input_x)
output_list = output.tolist()
print("Network output:", output_list)


#----- Minst Code ---- #

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

# Get the image you want to evaluate (e.g., image 19961)
image_index = 19961
image, _ = mnist_dataset[image_index]

# Flatten the image to match the input dimension of the network (784 neurons)
input_x_2 = image.view(-1).numpy()

# Evaluate the neural network with the input
output = my_network.evaluate(input_x_2)
output_list = output.tolist()

true_label = mnist_dataset[image_index][1]
pred_label = np.argmax(output_list)

# Compare the predicted label with the true label
if pred_label == true_label:
    print("Network correctly predicted the label:", true_label)
else:
    print("Network predicted:", pred_label, " but the true label is:", true_label)

# Print the neural network response
print("MINST output:", output_list)
