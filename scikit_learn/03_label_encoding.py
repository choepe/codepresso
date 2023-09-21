import seaborn as sns

# sklearn 의 LabelEncoder, OneHotEncoder 를 import 시키기
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#데이터 로딩
tips = sns.load_dataset('tips')
tips.head(5)
tips['sex'].unique()
# 인코딩할 컬럼 데이터 준비
items = tips['day']

# 1. 라벨인코딩(LabelEncoding) 실습
# LabelEncoder 객체 생성
encoder = LabelEncoder()

# fit 메소드에 인코딩할 데이터 전달
encoder.fit(items)

# transform 메소드를 통해 데이터 변환
labels = encoder.transform(items)

# 인코딩 결과 출력 (코드 제출시에는 주석 처리)
# print('Label Encoding Result:\n',labels)

# 인코딩된 수치형 데이터의 실제 클래스 확인 및 출력
classes = encoder.classes_
print('LabelEncoding classes:', classes)

# 디코딩 결과 확인 및 출력
inverse_result = encoder.inverse_transform([2])
# print('LabelDecoding result:', inverse_result)

# 2. 원핫인코딩(OneHotEncoding) 실습
# 2차원 데이터로 변환
labels = labels.reshape(-1,1)

# OneHotEncoder 객체 생성
one_hot_encoder = OneHotEncoder()

# .fit 메소드에 인코딩할 데이터 전달
one_hot_encoder.fit(labels)

# .transform 메소드를 통해 데이터 변환
one_hot_labels = one_hot_encoder.transform(labels)

# 인코딩 결과 출력 (코드 제출시에는 주석 처리)
# print('OneHotEncoding Result:\n', one_hot_labels.toarray())

# 속성 이용하여 인코딩된 데이터의 실제 클래스 확인
onehot_classes = one_hot_encoder.categories_
print('OneHotEncoding classes:', onehot_classes)
