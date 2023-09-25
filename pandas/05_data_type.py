import pandas as pd

# 실습 데이터 생성
score = {'name': ['Jessi', 'Emma', 'Alex', 'Jessi', 'Tom'],
         'age': [20, 24, 23, 20, 27],
         'score': [100, 95, 80, 85, 97],
         'grade': ['A', 'A', 'B', 'B', 'A'],
         'subject':['python', 'java', 'python', 'c', 'java']}

score_df = pd.DataFrame(data=score)
score_df.describe()
score_df[['grade', 'subject']].describe()
score_df.describe(include='all')

score_df['subject'].unique()
score_df['subject'].value_counts()
score_df['subject'].value_counts(normalize=True)