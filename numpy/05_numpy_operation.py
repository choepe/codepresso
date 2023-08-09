import numpy as np
import time

a = np.arange(1, 10).reshape(3, 3)
print("A---")
print(a)

b = np.arange(9, 0 , -1).reshape(3, 3)
print("B---")
print(b)

result = a + b
print("Result of A + B")
print(result)

result = np.add(a, b)
print("Result of np.add(a, b)")
print(result)

result = a / b
print("Result of A / B")
print(result)

result = np.divide(a, b)
print("Result of np.divide(a, b)")
print(result)

result = a > b
print("Result of comparison operator(>)")
print(result)

result = a != b
print("Result of comparison operator(!=)")
print(result)

python_arr = range(10000000)
start = time.time()
for i in python_arr:
    i + 1
stop = time.time()
print("Python(ms): ", (stop - start)*1000)

numpy_arr = np.arange(10000000)
start = time.time()
numpy_arr + 1
stop = time.time()
print("NumPy(ms): ", (stop - start)*1000)