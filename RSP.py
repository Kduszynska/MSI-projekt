import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.svm import LinearSVC
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from scipy.stats import mode

class RSP(BaseEnsemble, ClassifierMixin):
    def __init__(self, base_estimator=LinearSVC(), n_estimators=10, n_subspace_choose=10, n_subspace_features=20, hard_voting=True, random_state=None):
        #Klasyfikator bazowy
        self.base_estimator = base_estimator
        #liczba klasyfikatorów
        self.n_estimators = n_estimators
        #Liczba cech w jednej podprzestrzeni
        self.n_subspace_features = n_subspace_features
        #Tryb podejmowania decyzji
        self.hard_voting = hard_voting
        #ilość wybranych podprzestrzeni
        self.n_subspace_choose=n_subspace_choose
        #Ustawianie ziarna losowości
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, X, y):
        """Uczenie"""
        #Sprawdzenie czy X i y mają właściwy kształt
        X, y = check_X_y(X,y)
        #Przechowywanie nazw klas
        self.classes_ = np.unique(y)
        #Zapis liczby atrybutów
        self.n_features = X.shape[1]
        #Czy liczba cech w podprzestrzeni jest mniejsza od całkowitej liczby cech
        if self.n_subspace_features > self.n_features:
            raise ValueError("Number of features in subspace higher than number of features.")
        #Wylosowanie podprzestrzeni cech
        self.subspaces =[]
        #x = np.floor(len(X)/self.n_subspace_features)
        for i in range(self.n_estimators):
            try:
                self.subspaces.append(np.random.choice(len(X), size=self.n_subspace_features, replace=True))
            except:
                pass
        self.subspaces = np.array(self.subspaces)
       #Wybor podprzestrzeni przy pomocy prawdopodobieństwa a priori i odchylenia standardowego
        probability = []
        std_array = []
        for i in self.subspaces:
            y_new = y[i]
            l,c = np.unique(y_new, return_counts=True)
            for i in c:
                probability.append(c/self.n_subspace_features)
            std_array.append(np.std(probability))
        
        std_array= np.array(std_array)
        inc = np.argsort(std_array)
        part = inc[:self.n_subspace_choose]
        self.subspaces = self.subspaces[part]
        

        #Wyuczenie nowych modeli i stworzenie zespołu
        self.ensemble_ = []
        for i in range(self.n_estimators):
            self.ensemble_.append(clone(self.base_estimator).fit(X[self.subspaces[i]], y[self.subspaces[i]]))
        return self

    def predict(self, X):
        """Predykcja"""
        #Sprawdzenie czy modele są wyuczone
        check_is_fitted(self, "classes_")
        #sprawdzenie poprawności danych
        X = check_array(X)
        # Sprawdzenie czy liczba cech się zgadza
        if X.shape[1] != self.n_features:
            raise ValueError("number of features does not match")

        if self.hard_voting:
            # Podejmowanie decyzji na podstawie twardego glosowania
            pred_ = []
            # Modele w zespole dokonuja predykcji
            for i in range(self.n_estimators):
                pred_.append(self.ensemble_[i].predict(X))
            # Zamiana na miacierz numpy (ndarray)
            pred_ = np.array(pred_)
            # Liczenie glosow
            prediction = mode(pred_, axis=0)[0].flatten()
            # Zwrocenie predykcji calego zespolu
            return self.classes_[prediction]
        else:
            # Podejmowanie decyzji na podstawie wektorow wsparcia
            esm = self.ensemble_support_matrix(X)
            # Wyliczenie sredniej wartosci wsparcia
            average_support = np.mean(esm, axis=0)
            # Wskazanie etykiet
            prediction = np.argmax(average_support, axis=1)
            # Zwrocenie predykcji calego zespolu
            return self.classes_[prediction]

    def ensemble_support_matrix(self, X):
        # Wyliczenie macierzy wsparcia
        probas_ = []
        for i, member_clf in enumerate(self.ensemble_):
            probas_.append(member_clf.predict_proba(X[:, self.subspaces[i]]))
        return np.array(probas_)



