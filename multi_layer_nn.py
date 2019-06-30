import torch

def activation(x):
   return 1/(1+torch.exp(-x))
torch.manual_seed(7) # Set the random seed so things are predictable
features = torch.randn((1,5))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units 
n_output = 1                    # Number of output units

# Weights for inputs to hidden layer

W1 = torch.randn(n_input, n_hidden)

# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

h = activation(torch.mm(features, W1) + B1) 
output = activation(torch.mm(h, W2) + B2) 
print(output) 