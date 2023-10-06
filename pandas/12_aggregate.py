import pandas as pd
import numpy as np

# 실습 데이터 생성
score = {'sub1': [3, 9, 1, 1, 9],
         'sub2': [2, 9, np.nan, np.nan, 8],
         'sub3': [np.nan, 1, 5, 5, 7],
         'sub4': [np.nan, 3, np.nan, 1, np.nan]}

df = pd.DataFrame(data=score)


'''
count() 집계함수 실습
count 함수에 axis=1 인지 추가한 실습
sum() 집계함수 실습
'''
df.count()
df.count(axis=1)
df.sum()
df.sum(skipna=False)