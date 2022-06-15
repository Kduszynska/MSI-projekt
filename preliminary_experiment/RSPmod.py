import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from scipy.stats import mode
from sklearn.neighbors import KernelDensity


class RSPmod(BaseEnsemble, ClassifierMixin):
    """Zmodyfikowany Random Sample Partition - na cechach, modyfikacja polegająca na wyborze podprzestrzeni z jak najmniejszą gęstością """

    def __init__(self, base_estimator=LinearSVC(), n_estimators=10, n_subspace_choose=0.4, n_subspace_features=2, hard_voting=True, random_state=None):
        #Klasyfikator bazowy
        self.base_estimator = base_estimator
        #Liczba klasyfikatorów
        self.n_estimators = n_estimators
        #Liczba cech w jednej podprzestrzeni
        self.n_subspace_features = n_subspace_features
        #Tryb podejmowania decyzji
        self.hard_voting = hard_voting
        #Procent wybranych podprzestrzeni
        self.n_subspace_choose = n_subspace_choose
        #Ustawianie ziarna losowości
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, X, y):
        """Uczenie"""
        self.n_subspace_choose=1
        #Sprawdzenie czy X i y mają właściwy kształt
        X, y = check_X_y(X,y)
        #Przechowywanie nazw klas
        self.classes_ = np.unique(y)
        #Zapis liczby atrybutów
        self.n_features = X.shape[1]
        #Czy liczba cech w podprzestrzeni jest mniejsza od całkowitej liczby cech
        if self.n_subspace_features > self.n_features:
            raise ValueError("Number of features in subspace higher than number of features.")

        #Wiliczenie liczby stworzonych podprzestrzeni
        n_subspace = int(X.shape[1]/self.n_subspace_features)

        #Zamiana procentowej ilości wybieranych podprzestrzeni na liczbową 
        self.n_subspace_choose = int(self.n_subspace_choose * n_subspace)

        #Wylosowanie podprzestrzeni cech - bez zwracania
        self.subspaces =[]
        self.subspaces = np.random.choice(X.shape[1], size=(n_subspace, self.n_subspace_features), replace=False) 
        
        #Wybor podprzestrzeni na podstawie gęstości rozkładu
        log_density = []
        for i in self.subspaces:
            X_new =X[:,i]
            kde = KernelDensity(kernel='tophat', bandwidth=0.5).fit(X_new)
            log_density.append(kde.score(X_new))
        inc = np.argsort(log_density)
        part = inc[:self.n_subspace_choose]
        self.subspaces = self.subspaces[part]
        x = np.random.choice(n_subspace, size=(self.n_subspace_choose),replace=False)
        self.subspaces = self.subspaces[x,:]

        #Zmiana ilości estymatorów w zaleności od liczby wybranych podprzestrzeni
        if self.n_estimators > self.n_subspace_choose:
            self.n_estimators = self.n_subspace_choose
        
        #Wyuczenie nowych modeli i stworzenie zespołu
        self.ensemble_ = []
        for i in range(self.n_estimators):
            self.ensemble_.append(clone(self.base_estimator).fit(X[:,self.subspaces[i]], y))
        return self


    def predict(self, X):
        """Predykcja"""
        #Sprawdzenie czy modele są wyuczone
        check_is_fitted(self, "classes_")
        #Sprawdzenie poprawności danych
        X = check_array(X)
        #Sprawdzenie czy liczba cech się zgadza
        if X.shape[1] != self.n_features:
            raise ValueError("number of features does not match")


        if self.hard_voting:
            #Podejmowanie decyzji na podstawie twardego glosowania
            pred_ = []
            #Modele w zespole dokonuja predykcji
            for i in range(self.n_estimators):
                pred_.append(self.ensemble_[i].predict(X[:, self.subspaces[i]]))
            #Zamiana na miacierz numpy (ndarray)
            pred_ = np.array(pred_)
            #Liczenie glosow
            prediction = mode(pred_, axis=0)[0].flatten()
            #Zwrocenie predykcji calego zespolu
            return self.classes_[prediction]
        else:
            #Podejmowanie decyzji na podstawie wektorow wsparcia
            esm = self.ensemble_support_matrix(X)
            #Wyliczenie sredniej wartosci wsparcia
            average_support = np.mean(esm, axis=0)
            #Wskazanie etykiet
            prediction = np.argmax(average_support, axis=1)
            #Zwrocenie predykcji calego zespolu
            return self.classes_[prediction]

    def ensemble_support_matrix(self, X):
        """Wyliczenie macierzy wsparcia"""
        probas_ = []
        for i, member_clf in enumerate(self.ensemble_):
            probas_.append(member_clf.predict_proba(X[:, self.subspaces[i]]))
        return np.array(probas_)




