import pandas as pd

score = {'name': ['Jessi', 'Emma', 'Alex', 'Jessi', 'Tom'],
         'score': [100, 95, 80, 85, 97],
         'grade': ['A', 'A', 'B', 'B', 'A'],
         'subject':['python', 'java', 'python', 'c', 'java']}

score_df = pd.DataFrame(data=score)
score_df.info()
score_df.head(3)
score_df.tail(3)
score_df.sample(2, random_state=10)
score_df.sample(frac=0.5)