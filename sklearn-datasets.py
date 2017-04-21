__author__ = 'Aga'

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles, make_blobs, load_iris, load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

fig1 = plt.figure(figsize=(8, 8))
fig1.suptitle('Sklearn datasets')

#2 feat example
plt.subplot(321)
X, y = make_classification(n_samples=50,n_features=2, n_redundant=0, n_informative=2,
random_state=2, n_clusters_per_class=1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.title('Make-classification simple example', fontsize='small')


#make_moons
plt.subplot(322)
X, y = make_moons(noise = 0.1, random_state=1, n_samples=60)
plt.scatter(X[:,0],X[:,1], c = y, s = 100)
plt.title('Moons dataset', fontsize='small')


#circles
plt.subplot(323)
X, y = make_circles(n_samples=400, factor=.3, noise=.05)
plt.scatter(X[:,0],X[:,1], c = y, s = 100)
plt.title('Circle dataset', fontsize='small')


#random gaussians
plt.subplot(324)
plt.title("Three blobs", fontsize='small')
X1, Y1 = make_blobs(n_features=2, centers=3)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)

#digits
plt.subplot(325)
plt.title("Digits dataset", fontsize='small')
digits = load_digits()
data = scale(digits.data)
reduced_data = PCA(n_components=2).fit_transform(data)
y = digits.target
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y)


#iris
plt.subplot(326)
plt.title("Iris dataset", fontsize='small')
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
plt.scatter(X[:, 0], X[:, 1], c=y)

plt.show()
plt.close()


fig2 = plt.figure(figsize=(8, 8))
fig2.suptitle('Digit images')


for i in range(1,10):
    plt.subplot(int('33'+str(i)))
    plt.gray()
    plt.imshow(digits.images[i])
plt.show()
plt.close()
