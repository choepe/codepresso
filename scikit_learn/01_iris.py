import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

sklearn.__version__
iris = load_iris()
iris.DESCR

# 속성 이용하여 feature 확인(코드 제출시 주석 처리)
#print('Iris data shape:', iris.data.shape)
#print('Iris feature name\n:', iris.feature_names)
#print('Iris data\n:', iris.data)
#print('Iris data type\n:', type(iris.data))

# 속성 이용하여 class 확인 (코드 제출시 주석 처리)
#print('iris target name:\n',iris.target_names)
#print('iris target value:\n',iris.target)

# 데이터셋을 train, test 로 분할
# random_state 값은 강의와 동일하게 지정하세요.
x_train, x_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target,
                                                    test_size=0.3,
                                                    random_state=11)

# 분할된 데이터의 shape 확인 (코드 제출시 주석 처리)
#print('x_train.shape = ', x_train.shape)
#print('y_train.shape = ', y_train.shape)
#print('x_test.shape = ', x_test.shape)
#print('y_test.shape = ', y_test.shape)

# KNeighborsClassifier 의 객체 생성
knn = KNeighborsClassifier(n_neighbors=8)
type(knn)

# 훈련 데이터를 이용하여 분류 모델 학습
knn.fit(x_train, y_train)

# 학습된 knn 모델 기반 예측
y_pred = knn.predict(x_test)
print('Prediction:\n',y_pred)

# 모델 평가
score = knn.score(x_test, y_test)
print('Accuracy : {0:.5f}'.format(score))