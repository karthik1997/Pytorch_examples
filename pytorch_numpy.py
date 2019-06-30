import torch
import numpy as np
a = np.random.rand(4,3)
a




b = torch.from_numpy(a)
b

b.numpy()



# Multiply PyTorch Tensor by 2, in place
b.mul_(2)


# Numpy array matches new values from Tensor
a