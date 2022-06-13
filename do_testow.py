if __name__ == '__main__':
    dataset = 'ring'
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)


    n_splits = 5
    n_repeats = 10
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

    clf = RSP(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=15, hard_voting=True, random_state=123)
    scores = []
    for train, test in rskf.split(X, y):
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        scores.append(accuracy_score(y[test], y_pred))
    print("Hard voting - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))




    if __name__ == '__main__':
    dataset = 'ring'
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    print(X.shape)

    n_splits = 5
    n_repeats = 10
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

    clf = RSPmod(base_estimator=GaussianNB(), random_state=123)
    scores = []
    for train, test in rskf.split(X, y):
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        scores.append(accuracy_score(y[test], y_pred))
    print("Hard voting - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))
