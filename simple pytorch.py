import torch
def activation(x):
    """ Sigmoid activation function 
    
        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))
### Generate some data  
torch.manual_seed(7) # Set the random seed so things are predictable
features = torch.randn((1, 5)) # Features are 3 random normal variables
weights = torch.randn_like(features)  # True weights for our data, random normal variables again
bias = torch.randn((1, 1))
y = activation(torch.sum(features * weights) + bias)
print(y)
y = activation((features * weights).sum() + bias)

print(y)


y = activation(torch.mm(features, weights.view(5, 1)) + bias)
print(y)