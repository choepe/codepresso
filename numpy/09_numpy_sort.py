import numpy as np

sample = np.array([2, 5, 8, 1])
print("1D-Array Sample")
print(sample)

# 오름차순 정렬
result =np.sort(sample)
print("Result of np.sort")
print("Result: ", result)
print("Original: ", sample)

# 내림찬순 정렬
print("Test[::-1]")
print("Original: ", sample)
print("Result: ", result[::-1])

# 2D sample data
sample = np.array([[9, 11], [5, 1]])
print("2D-Array Sample")
print(sample)

result = np.sort(sample, axis=0)
print("np.sort(axis=0): ")
print(result)

result = np.sort(sample, axis=1)
print("np.sort(axis=1): ")
print(result)

name = np.array(['John', 'Samuel', 'Kate', 'Mike', 'Sarah'])
score = np.array([78, 84, 96, 88, 82])

# np.argsort()
sort_indexes = np.argsort(score)
print("np.argsort()")
print("sort indexes: ", sort_indexes)

print('sort values: ', name[sort_indexes])