import numpy as np

def printinfo(arr):
    print(arr)
    print(f"shape: {arr.shape}, size: {arr.size}, dtype: {arr.dtype}, dimension: {arr.ndim}")

# 샘플 데잍 생성
origin_arr = np.arange(15)
print("Origin ndarray")
printinfo(origin_arr)

# reshape()을 활용한 데이터 형상 변경
    # size는 동일해야 한다.
reshape_arr = np.reshape(origin_arr, (3, 5))
print("Reshape ndarray")
printinfo(reshape_arr)

# origin_arr에 저장된 1차원 배열을 -1을 활용하여 2차원으로 형상 변경
reshape_arr = origin_arr.reshape(-1, 1)
reshape_arr = origin_arr.reshape(1, -1)
print("Reshape(-1, N) to 2D array")
printinfo(reshape_arr)