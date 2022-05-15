from sklearn import datasets
from sklearn.ensemble import BaggingClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from RSM import RSM


""" Klasyfikatory uzyte w eksperymencie(?)
    GNB: GaussianNB
    SVC: SVC
    kNN: KNeighborsClassifier
    Linear SVC: LinearSVC"""

#zbiory danych
datasets = ['sonar']


#metody zespołowe
ensembles = {
    'none': None,
    'bagging': BaggingClassifier(base_estimator=GaussianNB()),
    'RSM': RSM(base_estimator=GaussianNB()),
}

n_datasets = len(datasets)
n_splits = 5
n_repeats = 2

#walidacja krzyzowa
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state = 44)

#tablice z wynikami
scores = np.zeros((len(ensembles), n_datasets, n_splits*n_repeats))

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=',')
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for ensemble_id, ensemble_name in enumerate(ensembles):
            
            X_train, y_train = X[train], y[train]
            
            #GNB
            clf_GNB = GaussianNB()
            clf_GNB.fit(X_train, y_train)
            y_pred = clf_GNB.predict(X[test])
            scores[ensemble_id, data_id, fold_id] = accuracy_score(y[test],y_pred)

            
#zapisanie  wyników 
np.save('results', scores)
