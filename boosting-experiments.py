__author__ = 'Aga'

from sklearn.datasets import load_breast_cancer, make_hastie_10_2, make_moons
from sklearn.model_selection import train_test_split
from adaboost import AdaBoost
from knn import getAccuracy
import warnings
warnings.filterwarnings("ignore")

def train(k, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = AdaBoost(k)

    (hyp, hyp_weights) = clf.fit(X_train, y_train)

    predictions = []
    for row in X_test:
        predictions.append(clf.predict(row, hyp, hyp_weights))

    print ("Accuracy for k = " + str(k) + " : " +str(getAccuracy(predictions, y_test)))

if __name__ == "__main__":
    """
    AdaBoost accuracy vs the number of iterations
    Datasets: breast_cancer, make_hastie_10_2, make_moons
    """
    for k in range(5, 100, 10):
        data = load_breast_cancer()
        X = data.data
        y = data.target
        y[y == 0] = -1

        train(k, X, y)

        X, y = make_hastie_10_2(n_samples=2000)
        train(k, X, y)

        X, y = make_moons(noise = 0.1, random_state=1, n_samples=2000)
        train(k, X,y)



