import pandas as pd
import numpy as np

# 실습 데이터 생성
df = pd.DataFrame(data = np.arange(18).reshape(6,3),
				  index = ['a','b','c','d','e','f'],
                  columns=['col1','col2','col3'])

df['col4'] = pd.Series(data = [1.7, 1.2, 2.4],
                       index = ['a','e','c'])
df.loc['c'] = None
df.info()
df.isna()
df.isna().sum()
df.isna().sum(axis=1)

df.dropna()
df.dropna(how='all')
df.dropna(how='all', inplace=True)
df.dropna(axis='columns')

df.iloc[:2, 0] = np.nan
df.iloc[:4, 1] = np.nan
df.fillna(0)

replace_set = {'col2': 0, 'col4': '100'}
df.fillna(replace_set)

replace_set = {'col1': df['col1'].mean()}
df.fillna(replace_set)