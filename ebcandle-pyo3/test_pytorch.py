import ebcandle
import torch

# convert from ebcandle tensor to torch tensor
t = ebcandle.randn((3, 512, 512))
torch_tensor = t.to_torch()
print(torch_tensor)
print(type(torch_tensor))

# convert from torch tensor to ebcandle tensor
t = torch.randn((3, 512, 512))
ebcandle_tensor = ebcandle.Tensor(t)
print(ebcandle_tensor)
print(type(ebcandle_tensor))
