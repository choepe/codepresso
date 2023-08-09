import numpy as np

sample = np.arange(1, 10).reshape(3, 3)
print("Sample")
print(sample)

# axis=None (default)
print('sum :', sample.sum())
print('mean :', sample.mean())
print('std :', sample.std())
print('max :', sample.max())
print('min :', sample.min())

# axis=0 (default)
print('sum :', sample.sum(axis=0))
print('mean :', sample.mean(axis=0))
print('std :', sample.std(axis=0))
print('max :', sample.max(axis=0))
print('min :', sample.min(axis=0))

# axis=1 (default)
print('sum :', sample.sum(axis=1))
print('mean :', sample.mean(axis=1))
print('std :', sample.std(axis=1))
print('max :', sample.max(axis=1))
print('min :', sample.min(axis=1))