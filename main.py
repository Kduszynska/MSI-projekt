from sklearn import datasets
from sklearn.ensemble import BaggingClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from RSM import RSM
from RSP import RSP
from RSPmod import RSPmod


""" Klasyfikatory uzyte w eksperymencie(?)
    GNB: GaussianNB
    SVC: SVC
    kNN: KNeighborsClassifier
    Linear SVC: LinearSVC"""

#zbiory danych
datasets = ['arcene','BurkittLymphoma','kits-subset','rsctc2010_2']


#base_estimator1=GaussianNB()
#base_estimator2=LinearSVC(random_state=1234)
#base_estimator3=SVC(random_state=1234)
base_estimator4=KNeighborsClassifier()
n_estimators=2
#metody zespołowe
ensembles = {
    #'GNB': GaussianNB(),
    #'LSVC': LinearSVC(random_state=1234),
    #'SVC': SVC(random_state=1234),
    'kNN': KNeighborsClassifier(),
    'bagging': BaggingClassifier(base_estimator=base_estimator4, n_estimators=n_estimators),
    'RSM': RSM(base_estimator=base_estimator4, n_estimators=n_estimators, n_subspace_features=2),
    'RSP': RSP(base_estimator=base_estimator4, n_estimators=n_estimators, n_subspace_choose=0.6, n_subspace_features=2),
    'RSPmod': RSPmod(base_estimator=base_estimator4, n_estimators=n_estimators, n_subspace_choose=0.6, n_subspace_features=2),
}

if __name__ == '__main__':

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
                ENS = ensembles[ensemble_name].fit(X_train, y_train)
                y_pred = ENS.predict(X[test])
                scores[ensemble_id, data_id, fold_id] = accuracy_score(y[test],y_pred)
            
    #zapisanie  wyników 
    np.save('results', scores)

    print(np.mean(scores,axis=2))