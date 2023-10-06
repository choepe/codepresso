import pandas as pd

# 실습 데이터 생성
d = {'name': ['Jessi', 'Emma', 'Alex', 'Jessi', 'Tom'],
     'score': [100, 95, 80, 85, 97],
     'grade': ['A', 'A', 'B', 'B', 'A'],
     'subject':['python', 'java', 'python', 'c', 'java']}

sample_df = pd.DataFrame(data=d)
sample_df.set_index('name', inplace=True)

'''
start
end
step
'''
subset = sample_df[1:4]
subset = sample_df[::2]
subset = sample_df[:'Alex']


'''
loc 프로퍼티 활용한 슬라이싱 실습
주석(#)을 지우고, 주어진 빈칸의 코드를 완성하세요.
'''
subset = sample_df.loc[: 'Emma', ['subject', 'grade']]
subset = sample_df.iloc[:4, -1]