from torch import nn

class network(nn.Module):
  def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 128)
        # Hidden to hidden layer linear transformation
        self.hidden_1 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(64, 10)
        # Define sigmoid activation and softmax output 
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
      
      
def forward(self, x):
      x = self.hidden(x)
      x = self.hidden_1(x)
      x = self.Relu(x)
      x = self.Softmax(x)
      
      
      return x


model = network()
model


print(model.hidden.weight)
print(model.hidden.bias)

model.hidden.bias.data.fill_(0)

# sample from random normal with standard dev = 0.01
model.hidden.weight.data.normal_(std=0.01)
