import numpy as np

# dtype 인자 지정을 통한 데이터 타입 변경
data = [1, 2, 3]
cpx_arr = np.array(data, dtype=complex)
str_arr = np.array(data, dtype=str)

print("Complex: ", cpx_arr)
print("String: ", str_arr)

# 샘플 데이터 생성
origin = np.arange(1, 2, 0.2)
print("Original data")
print(origin)
print("dtype: ", origin.dtype)

# astype(dtype)을 활용한 데이터 타입 변경
result = origin.astype(int)
print("Result of astype(int)")
print(result)

print("Original data")
print(origin)