__author__ = 'Aga'

"""
Ozone peak prediction using SVM-RBF
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

input_file = "ozone.csv"
df = pd.read_csv(input_file, sep=' ')

df.station = pd.get_dummies(df.station)

scaler = StandardScaler()

df[["MOCAGE", "TEMPE", "NO2", "RMH2O", "VentMOD"]] = scaler.fit_transform(df[["MOCAGE", "TEMPE", "NO2", "RMH2O", "VentMOD"]])

#ozone peak - an O3obs value greater than 150

df.loc[df["O3obs"] > 150, 'y'] = 1
df.loc[df["O3obs"] <= 150, 'y'] = 0

y = df["y"]
X = df.ix[:, df.columns != 'y']


steps = [('scaler', StandardScaler()), ('rbf_svm', SVC())]
pipeline = Pipeline(steps)

C_range = [2**exp for exp in range(-6,16, 2)]
gamma_range = [2**exp for exp in range(-16, 3, 2)] #Practical Guide to SVC
seed = 777
test_size = 0.3
ss_test_size = 0.3
score = 'accuracy'

param_grid = dict(rbf_svm__gamma=gamma_range, rbf_svm__C=C_range)


X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

ss = ShuffleSplit(test_size=ss_test_size, random_state=seed)

grid = GridSearchCV(pipeline, param_grid, cv=ss, scoring=score)

grid.fit(X_train, y_train)

print("Grid scores on development set:")
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))

print("Best parameters set found on development set:")
print(grid.best_params_)
print("Accuracy of the best estimator: ")
print(grid.best_score_)

print ("Evaluation on the test set:")
y_true, y_pred = y_test, grid.predict(X_test)
print(classification_report(y_true, y_pred))

print("Plot grid scores for every iteration: ")
scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))
