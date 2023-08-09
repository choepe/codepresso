import numpy as np
# as : alias

print(np.__version__)

# 1차원 ndarray
data = [10, 20, 30]
arr = np.array(data)

print('Create 1D Array')
print(arr)
print('type = ', type(arr))
print("shape: {}, size: {}, dtype: {}, dimension: {}".format(arr.shape, arr.size, arr.dtype, arr.ndim))
print(f"shape: {arr.shape}, size: {arr.size}, dtype: {arr.dtype}, dimension: {arr.ndim}")

# 2차원 ndarray
data = [[1, 2, 3], [4, 5, 6]]
arr = np.array(data, dtype=float)

print("Create 2D Array")
print(arr)
print("type = ", type(arr))
print(f"shape: {arr.shape}, size: {arr.size}, dtype: {arr.dtype}, dimension: {arr.ndim}")

# 3차원 ndarray
data = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
arr = np.array(data, dtype=float)

print("Create 3D Array")
print(arr)
print("type = ", type(arr))
print(f"shape: {arr.shape}, size: {arr.size}, dtype: {arr.dtype}, dimension: {arr.ndim}")