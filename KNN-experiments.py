from sklearn.datasets import make_classification, make_moons, make_circles
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from knn import getLabel, getAccuracy, getNeighbors


"""
KNN - how accuracy changes with growing K
I used 3 datasets of size 100 and used on them the algorithm implemented above.
In each case, we can observe the fall of accuracy for K approaching the size of the dataset.
K between 5 and 10 is enough for a good results on these small datasets.
"""
if __name__ == "__main__":
	fig = plt.figure(figsize=(8, 8))
	fig.suptitle('Accuracy vs K in KNN - algorithm implemented from scratch')

	plt.subplot(311)
	X, y = make_classification(n_samples=100,n_features=2, n_redundant=0, n_informative=2,
	random_state=2, n_clusters_per_class=1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	data = np.column_stack([X_train,y_train])

	ks = range(1,data.shape[0])
	acc = []

	for k in ks:
		predictions = []
		for x in X_test:
			predictions.append(getLabel(getNeighbors(data, x, k)))
		acc.append(getAccuracy(y_test, predictions))

	plt.plot(ks, acc)



	plt.title("Make_classification", fontsize="small")

	plt.subplot(312)
	X, y = make_moons(noise = 0.1, random_state=1, n_samples=100)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	data = np.column_stack([X_train,y_train])

	ks = range(1,data.shape[0])
	acc = []

	for k in ks:
		predictions = []
		for x in X_test:
			predictions.append(getLabel(getNeighbors(data, x, k)))
		acc.append(getAccuracy(y_test, predictions))

	plt.plot(ks, acc)
	plt.title("Make_moons", fontsize="small")

	plt.subplot(313)
	X, y = make_circles(n_samples=100, factor=.3, noise=.05)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	data = np.column_stack([X_train,y_train])

	ks = range(1,data.shape[0])
	acc = []

	for k in ks:
		predictions = []
		for x in X_test:
			predictions.append(getLabel(getNeighbors(data, x, k)))
		acc.append(getAccuracy(y_test, predictions))

	plt.plot(ks, acc)
	plt.title("Make_circles", fontsize="small")

	plt.show()
	plt.close()

"""
Frontier
"""

fig = plt.figure(figsize=(8, 8))
fig.suptitle('Evolution of the frontier - make classification')

X, y = make_classification(n_samples=100,n_features=2, n_redundant=0, n_informative=2,
	random_state=2, n_clusters_per_class=1)

h = .02 # grid step
x_min= X[:, 0].min() - 1
x_max= X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))


for k in range(1, X.shape[0], 10):
	clf = KNeighborsClassifier(n_neighbors=k)
	clf.fit(X, y)

	Z2d = clf.predict(np.c_[xx.ravel(),yy.ravel()])
	Z2d=Z2d.reshape(xx.shape)
	plt.subplot('33'+str((k-1)/10))
	plt.pcolormesh(xx,yy,Z2d, cmap=plt.cm.Paired)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.show()
plt.close()

fig = plt.figure(figsize=(8, 8))
fig.suptitle('Evolution of the frontier - make moons')

X, y = make_moons(noise = 0.1, random_state=1, n_samples=100)

h = .02 # grid step
x_min= X[:, 0].min() - 1
x_max= X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))


for k in range(1, X.shape[0], 10):
	clf = KNeighborsClassifier(n_neighbors=k)
	clf.fit(X, y)

	Z2d = clf.predict(np.c_[xx.ravel(),yy.ravel()])
	Z2d=Z2d.reshape(xx.shape)
	plt.subplot('33'+str((k-1)/10))
	plt.pcolormesh(xx,yy,Z2d, cmap=plt.cm.Paired)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.show()
plt.close()

fig = plt.figure(figsize=(8, 8))
fig.suptitle('Evolution of the frontier - make circles')

X, y = make_circles(n_samples=100, factor=.3, noise=.05)

h = .02 # grid step
x_min= X[:, 0].min() - 1
x_max= X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))


for k in range(1, X.shape[0], 10):
	clf = KNeighborsClassifier(n_neighbors=k)
	clf.fit(X, y)

	Z2d = clf.predict(np.c_[xx.ravel(),yy.ravel()])
	Z2d=Z2d.reshape(xx.shape)
	plt.subplot('33'+str((k-1)/10))
	plt.pcolormesh(xx,yy,Z2d, cmap=plt.cm.Paired)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.show()
plt.close()


