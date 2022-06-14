import numpy as np
from RSP import RSP
from RSPmod import RSPmod
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

"""Eksperyment wstępny"""

n_estimators=2

#PROCENT WYBRANYCH PODPRZESTRZENI
v = [x/10 for x in range(1,10)]

#KLASYFIKATORY BAZOWE
base_estimators=['GNB', 'LSVC', 'SVC', 'kNN']
base_estimator=[GaussianNB(), LinearSVC(random_state=1234), SVC(random_state=1234), KNeighborsClassifier()]

#METODY ZESPOŁOWE
ensembles = {}
for i in range(9):
    ensembles[f"RSP {v[i]}"]=RSP(base_estimator=base_estimator[3], n_estimators=n_estimators, n_subspace_choose=v[i], n_subspace_features=2)
    ensembles[f"RSPmod {v[i]}"]=RSPmod(base_estimator=base_estimator[3], n_estimators=n_estimators, n_subspace_choose=v[i], n_subspace_features=2)

if __name__ == '__main__':

    #WALIDACJA KRZYŻOWA
    n_splits = 5
    n_repeats = 2
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state = 44)

    #WYNIKI - tablica
    scores = np.zeros((len(ensembles), n_splits*n_repeats))

    #SYNTETYCZNY ZBIOR
    X, y = datasets.make_classification(n_samples=5000, n_features=20, n_classes=4, n_informative=20, n_redundant=0, flip_y=0.05, n_clusters_per_class=1)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for ensemble_id, ensemble_name in enumerate(ensembles):
            X_train, y_train = X[train], y[train]
            ENS = ensembles[ensemble_name].fit(X_train, y_train)
            y_pred = ENS.predict(X[test])
            scores[ensemble_id, fold_id] = accuracy_score(y[test],y_pred)
            
    #ZAPISANIE WYNIKOW
    np.save('results_preliminary_experiment', scores)

