import numpy as np

# range((start), stop, (step))
def printinfo(arr):
    print(arr)
    print(f"shape: {arr.shape}, size: {arr.size}, dtype: {arr.dtype}, dimension: {arr.ndim}")

# np.arange()를 통한 ndarray 생성
arr = np.arange(2, 10)
print("Create ndarray by np.arange()")
print(arr)
printinfo(arr)

# np.zeros(shape, dtype=float)
    # np.zeros()를 통한 ndarray 생성
arr = np.zeros(5)
print("Create ndarray by np.zeros()")
printinfo(arr)

arr = np.zeros((3,4), dtype=int)
print("Create ndarray by np.zeros()")
printinfo(arr)

# np.ones(shape, dtype=float)
    # np.ones()를 통한 ndarray 생성
arr = np.ones((3,4))
print("Create ndarray by np.ones()")
printinfo(arr)

# np.full(shape,fill_value, dtype=float)
    # np.full()를 통한 ndarray 생성
arr = np.full((3,4), 77)
print("Create ndarray by np.full()")
printinfo(arr)

# np.empty(shape)
arr = np.empty((2,2))
print("Create ndarray by np.empty()")
printinfo(arr)

# *_like: shape 가져옴
data = [[1, 2, 3], [4, 5, 6]]
sample = np.array(data)
print("sample")
printinfo(sample)

one_array = np.ones_like(data, dtype=float)
print("Create ndarray by np.ones_like()")
printinfo(one_array)