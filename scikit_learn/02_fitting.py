import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target,
                                                    test_size=0.3,
                                                    random_state=11)

k_list = range(1, 100, 2)
acc_train = []
acc_test = []

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    acc_train.append(knn.score(x_train, y_train))
    acc_test.append(knn.score(x_test, y_test))

plt.figure(figsize=(10, 4))
plt.plot(k_list, acc_train, 'b--')
plt.plot(k_list, acc_test, 'g')
plt.title("Training and Test Accuracy")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.0)
plt.show()