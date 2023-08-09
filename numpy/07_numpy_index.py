import numpy as np

# 1D array sample
sample = np.arange(5, 15)
print("1D-Array Sample: ", sample)
print('sample[3] = ', sample[3])
sample[3] = 0
print('1Dj-Array Sample: ', sample)

# 2D array sample
sample = np.arange(16).reshape((4, 4))
print("2D-Array Indexing: \n", sample)
print('sample[2, 3] = ', sample[2, 3])

sample[2, 3] = 0
print(sample)

# 3D array sample
sample = np.arange(16).reshape((4, 2, 2))
print("3D-Array Indexing: \n", sample)
print('sample[1, 1, 1] = ', sample[1, 1, 1])