import pandas as pd

url = 'https://codepresso-online-platform-public.s3.ap-northeast-2.amazonaws.com/learning-resourse/titanic_train.csv'
titanic_df = pd.read_csv(url)
type(titanic_df)
titanic_df.info()
titanic_df.head()
titanic_df.tail()
titanic_df.sample()

# students_score.xlsx 이 저장된 url
url = 'https://codepresso-online-platform-public.s3.ap-northeast-2.amazonaws.com/learning-resourse/python-data-analysis/students_score.xlsx'
sample_df = pd.read_excel(url)


'''
1) header=none 옵션으로 컬럼 없이 데이터 읽기
2) df.columns 에 새로운 컬럼명 저장하기
주석(#)을 지우고, 주어진 빈칸의 코드를 완성하세요.
'''
sample_df = pd.read_excel(url, header=None)
sample_df.columns = ['name', 'age', 'score', 'grade', 'subject']
sample_df

sample_df = pd.read_excel(url, header=2)
sample_df.columns = ['name', 'age', 'score', 'grade', 'subject']
sample_df

'''
'students' sheet 에 있는 데이터 읽기
주석(#)을 지우고, 주어진 빈칸의 코드를 완성하세요.
'''
student_df = pd.read_excel(url, sheet_name='students')