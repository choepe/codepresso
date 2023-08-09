import torch
import numpy as np

print(torch.__version__)
# shape=(2, 2, 3)
d3_tensor = torch.Tensor([[[1, 2, 3],
                           [4, 5, 6]],
                          [[-1, -2, -3],
                           [-4, -5, -6]]])

d3_tensor
d3_tensor.ndim
d3_tensor.shape
d3_tensor.dtype

# shape=(2, 2, 3), value=0
d3_zeros_tensor = torch.zeros(size=(2, 2, 3))
d3_zeros_tensor
d3_zeros_tensor.ndim
d3_zeros_tensor.shape
d3_zeros_tensor.dtype

# shape=(2, 2, 3), value=1
d3_ones_tensor = torch.ones(size=(2, 2, 3))
d3_ones_tensor
d3_ones_tensor.ndim
d3_ones_tensor.shape
d3_ones_tensor.dtype

# shape=(2, 2, 3), value=randum
d3_rand_tensor = torch.rand(size=(2, 2, 3))
d3_rand_tensor
d3_rand_tensor.ndim
d3_rand_tensor.shape
d3_rand_tensor.dtype

d3_np_tensor = np.array([[[1, 2, 3],
                          [4, 5, 6]],
                         [[-1, -2, -3],
                          [-4, -5, -6]]])
d3_torch_tensor = torch.Tensor(d3_np_tensor)
d3_torch_tensor = torch.from_numpy(d3_np_tensor)