import numpy as np

# 1D sample data
sample = np.arange(5, 15)
print("1D-Array Sample")
print(sample)

print("Slicing")
print(sample[2:5])
print(sample[:5])
print(sample[7:])
print(sample[2:-1])
print(sample[:])

# 2D sample data
sample = np.arange(1, 17).reshape(4, 4)
print("2D-Array Sample")
print(sample)

print("Slicing")
print(sample[1:3, 1:3])
print(sample[:3, 0:1])
print(sample[3:, :])
print(sample[:, 2])
print(sample[:, :])

print("Slicing by Negative")
print(sample[:, 1:-1])
print(sample[:-1, -2])
print(sample[:, -1])
