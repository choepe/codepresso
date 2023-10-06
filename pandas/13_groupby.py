import pandas as pd
import numpy as np

# 실습 데이터 생성
sample = {'product':['a','b','a','b','a','b','a','a'],
          'sensor':['s1','s1','s2','s3','s2','s2','s1','s3'],
          'x':np.arange(1,9),
          'y':np.arange(5,13)}

df = pd.DataFrame(data=sample)


'''
slit: 전체 데이터를 그룹 별로 나눔
apply: 각 그룹별로 집계함수를 적용
combine: 그룹별 집계 결과를 하나로 합침
DataFrameGroupBy: 그룹화된 데이터를 key-value 형태로 저장
'''
grouped_product = df.groupby('product')
for key, value in grouped_product:
      print("-----------------------")
      print("key :", key)
      print("value :\n", value)

#for key in grouped_product:
#      print("-----------------------")
#      print("key :", key)

grouped_product = df.groupby('product').sum()
grouped_product = df.groupby(['product', 'sensor']).sum()
grouped_product = df.groupby(['product', 'sensor'])['x'].sum()

condition = {'x': 'max', 'y': 'min'}
grouped_product = df.groupby(['product', 'sensor']).agg(condition)