import pandas as pd
import numpy as np

'''
주어진 series 데이터를 이용하여 DataFame 을 생성하세요.
주석(#)을 지우고, 주어진 빈칸의 코드를 완성하세요.
'''
series = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
          'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

result = pd.DataFrame(data=series)


'''
딕셔너리가 저장된 리스트 객체(data)를 이용하여
index 가 있는 DataFrame 을 생성하세요.
주석(#)을 지우고, 주어진 빈칸의 코드를 완성하세요.
'''
data = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
idx = ['row1', 'row2']

result = pd.DataFrame(data, index=idx)


'''
주어진 ndarray 데이터를 이용하여
index 와  column이 지정된 DataFrame 을 생성하세요.
주석(#)을 지우고, 주어진 빈칸의 코드를 완성하세요.
'''
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
col = ['col1', 'col2', 'col3']
idx = ['row1', 'row2', 'row2']

result = pd.DataFrame(arr, columns=col, index=idx)