import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pprint import pprint
from collections import Counter


class KNN:
    def __init__(self, k_neighbors=3):
        self.k_neighbors = k_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted = [self._predict(x) for x in X]
        return np.array(predicted)

    def _predict(self, x):
        # compute distances
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k_neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # get majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum(x1 - x2) ** 2)

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
print("X Train data size - " + str(X_train.shape))
print("First sample - " + str(X_train[0]))

print("Y Train data size - " + str(y_train.shape))
print("Train labels - " + str(y_train))

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k', s=20)
plt.show()

knn = KNN(5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

accuracy = np.sum(predictions == y_test) / len(predictions)
print("Accuracy = " + str(accuracy))