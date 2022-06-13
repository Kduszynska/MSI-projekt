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

#zbiór Flora openml.org (99x15000)

""" Klasyfikatory uzyte w eksperymencie(?)
    GNB: GaussianNB
    SVC: SVC
    kNN: KNeighborsClassifier
    Linear SVC: LinearSVC"""



n_estimators=2
#metody zespołowe
v = [x/10 for x in range(1,10)]
base_estimator=[GaussianNB(), LinearSVC(random_state=1234), SVC(random_state=1234), KNeighborsClassifier()]
ensembles = {}
for i in range(9):
    for j in range(4):
        ensembles[f"RSP{ v[i]}"]=RSP(base_estimator=base_estimator[j], n_estimators=n_estimators, n_subspace_choose=v[i], n_subspace_features=2)
        ensembles[f"RSPmod{ v[i]}"]=RSPmod(base_estimator=base_estimator[j], n_estimators=n_estimators, n_subspace_choose=v[i], n_subspace_features=2)
if __name__ == '__main__':

    
    n_splits = 5
    n_repeats = 2

    #walidacja krzyzowa
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state = 44)

    #tablice z wynikami
    scores = np.zeros((len(ensembles), n_splits*n_repeats))

    X, y = datasets.make_classification(n_samples=5000, n_features=20, n_classes=4, n_informative=20, n_redundant=0, flip_y=0.05, n_clusters_per_class=1)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for ensemble_id, ensemble_name in enumerate(ensembles):
            X_train, y_train = X[train], y[train]
            ENS = ensembles[ensemble_name].fit(X_train, y_train)
            y_pred = ENS.predict(X[test])
            scores[ensemble_id, fold_id] = accuracy_score(y[test],y_pred)
            
    #zapisanie  wyników 
    np.save('results', scores)

    print(np.mean(scores,axis=1))