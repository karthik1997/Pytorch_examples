import torch

from torchvision import datasets, transforms
import helper
# Define a transform to normalize the data

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5, ), (0.5, ))])

# Download and load the training data

trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download = True, train = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)

# Download and load the testing data

testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download = True, train = False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle =True)


image, label = next(iter(trainloader))

from torch import nn, optim
import torch.nn.functional as F

class Classifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(784, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 64)
    self.fc4 = nn.Linear(64, 10)
    
    
  def forward(self, x):
     # make sure input tensor is flattened
     x = x.view(x.shape[0], -1)
     x = F.relu(self.fc1(x))
     x = F.relu(self.fc2(x))
     x = F.relu(self.fc3(x))
     x = F.log_softmax(self.fc4(x), dim=1) 
    
     return x
    
model = Classifier()
criterian = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 5

for e in range(epochs):
  running_loss = 0
  for images, labels in trainloader:
    logits = model(images)
    loss = criterian(logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()
  else:
        print(f"Training loss: {running_loss}")

      
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import helper


# Test out your network!

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[1]

# TODO: Calculate the class probabilities (softmax) for img
ps = torch.exp(model(img))

# Plot the image and probabilities
helper.view_classify(img, ps, version='Fashion')
