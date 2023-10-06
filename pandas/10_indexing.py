import pandas as pd

# 실습 데이터 생성
d = {'name': ['Jessi', 'Emma', 'Alex', 'Jessi', 'Tom'],
     'score': [100, 95, 80, 85, 97],
     'grade': ['A', 'A', 'B', 'B', 'A'],
     'subject':['python', 'java', 'python', 'c', 'java']}

df = pd.DataFrame(data=d)
df.set_index('name', inplace=True)

'''
loc[]: 인덱스 명(값)을 기준으로 행 데이터 조회(Location)
iloc[]: 인덱스 번호(position)를 기준으로 행 데이터 조회(Integer Location)
'''
subset = df.loc[['Jessi', 'Emma']] # row index values
subset = df.iloc[[0, 4]] # row index position

'''
Boolean Indexing
비교연산(>, >=, <, <=, ==, !=)
Boolean 값 을 활용
'''
subset = df.loc[df['score'] >= 95] # columnindex values
subset = df.loc[df['subject'] == 'python']