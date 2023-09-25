import pandas as pd

# 실습 데이터 생성
score = {'name': ['Jessi', 'Emma', 'Alex', 'Jessi', 'Tom'],
         'score': [100, 95, 80, 85, 97],
         'grade': ['A', 'A', 'B', 'B', 'A'],
         'subject':['python', 'java', 'python', 'c', 'java']}
c = ['name', 'subject', 'score', 'grade', 'etc']

df = pd.DataFrame(data=score, columns=c)

semester_data = pd.Series(['20-01', '20-01', '20-02', '20-01'])
df['semester'] = semester_data

df['high_score'] = df['score'] > 90

df.loc[6] = ['Jina', 'python', 100, 'A', 1, '20-02', True]

df.drop(6)
df.drop(6, inplace=True)

df.drop(columns=['etc'], inplace=True)