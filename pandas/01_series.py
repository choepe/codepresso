import pandas as pd
pd.__version__

year = ['2019', '2020', '2021', '2022']
result = pd.Series(data=year)
type(result)

result.index
result.values
result.dtype
result.shape

result.name = 'Year'
result.index.name = 'No.'

idx = ['a', 'b', 'c', 'd']
result = pd.Series(data=year, index=idx, name='Year')

score = {'Kim': 85, 'Han': 89, 'Lee': 99, 'Choi': 70}
result = pd.Series(data=score, name='Score')