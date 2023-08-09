import numpy as np


d0_tensor = np.array(3)
d0_tensor
d0_tensor.ndim

d1_tensor = np.array([1., 2., 3.,])
d1_tensor
d1_tensor.ndim

d2_tensor = np.array([[1, 2, 3],
                      [4, 5, 6]])
d2_tensor
d2_tensor.ndim

d3_tensor = np.array([[[1, 2, 3],
                       [4, 5, 6]],
                      [[-1, -2, -3],
                       [-4, -5, -6]]])
d3_tensor
d3_tensor.ndim

print(d0_tensor.ndim)
print(d1_tensor.ndim)
print(d2_tensor.ndim)
print(d3_tensor.ndim)

print(d0_tensor)
d0_tensor.shape

print(d1_tensor)
d1_tensor.shape

print(d2_tensor)
d2_tensor.shape

print(d3_tensor)
d3_tensor.shape
d3_tensor.ndim
d3_tensor.size
d3_tensor.dtype