import pandas as pd

# 실습 데이터 생성
score = {'name': ['Jessi', 'Emma', 'Alex', 'Jessi', 'Tom'],
         'score': [100, 95, 80, 85, 97],
         'grade': ['A', 'A', 'B', 'B', 'A'],
         'subject':['python', 'java', 'python', 'c', 'java']}
c = ['name', 'subject', 'score', 'grade', 'etc']

df = pd.DataFrame(data=score, columns=c)


'''
인덱싱 기법과 . 연산자로 열 데이터 조회 실습
'''
# 인덱싱 기법
df['name']
type(df['name'])
# . 연산자
df.name
type(df.name)

df[['name', 'subject', 'grade']]
type(df[['name', 'subject', 'grade']])

df['etc'] = 0
df

df.loc[1]
df.loc[[0, 2, 4]]

df.loc[0] = ['Jessi', 'java', 70, 'C', 1]

row_idx = [1,2,4]
col_idx = ['name', 'subject', 'grade']
df.loc[row_idx, col_idx]