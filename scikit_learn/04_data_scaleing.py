'''
-------- [최종 출력 결과] --------
Average
temp         ***
atemp        ***
humidity     ***
windspeed    ***
dtype: float64
Variance
temp          ***
atemp         ***
humidity     ***
windspeed     ***
dtype: float64
--------- StandardScaler ---------
Average
temp         ***
atemp       ***
humidity    ***
windspeed   ***
dtype: float64
Variance
temp         ***
atemp        ***
humidity     ***
windspeed    ***
dtype: float64
----------------------------------
'''
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# URL 통해서 캐글의 자전거 대여 수요 데이터셋 다운로드
url = 'https://codepresso-online-platform-public.s3.ap-northeast-2.amazonaws.com/learning-resourse/python-machine-learning-20210326/bike-demand.csv'
df_bike = pd.read_csv(url)
df_bike.head(5)
df_bike[['temp', 'atemp', 'humidity', 'windspeed']]
# temp, atemp, humidity, windspeed	컬럼 데이터만 저장
df_bike_num = df_bike.iloc[:, 5:9]
df_bike_num.head(5)
# 코드 제출시 아래 코드는 주석 처리 필요
df_bike_num.head(5)

# 각 컬럼별 평균, 분산 출력
print('Average')
print(np.round_(df_bike_num.mean(),3))
print('Variance')
print(np.round_(df_bike_num.var(),3))

# StandardScaler 객체 생성
scaler = StandardScaler()

# StandardScaler 모델 통해 데이터 분포 분석
scaler.fit(df_bike_num)

# 모델 통해서 데이터 스케일링 후 반환
result = scaler.transform(df_bike_num)

# 스케일된 결과 데이터를 DataFrame 으로 저장
scaled_bike = pd.DataFrame(data=result,
                           columns=df_bike_num.columns)

# 각 컬럼별 평균, 분산 출력
print('--------- StandardScaler ---------')
print('Average')
print(np.round_(scaled_bike.mean(),3))
print('Variance')
print(np.round_(scaled_bike.var(),3))

# 박스플로(boxplot) 으로 시각화
plt.figure(figsize=(10,6))
scaled_bike.boxplot(column=['temp', 'atemp', 'humidity', 'windspeed'])
plt.show()