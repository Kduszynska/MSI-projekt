from matplotlib.pyplot import axis
import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_ind
from eksperyment_wstepny import ensembles, datasets

# Zmienne globalne uzyte w testach statystycznych
alpha=.05
m_fmt="%.3f"
std_fmt="%.3f"
nc="---"
db_fmt="%s"
tablefmt="plain"
clfs=ensembles
n_clfs = len(clfs)
headers = list(clfs.keys())

if __name__=="__main__":
        # Pobranie wyników
        scores = np.load("results.npy")
        mean_scores = np.mean(scores, axis=1)
        stds = np.std(scores, axis=1)
        t = []

        # Wiersz z wartoscia srednia
        t.append([m_fmt % v for v in mean_scores[:]])
        # Jesli podamy std_fmt w zmiennych globalnych zostanie do tabeli dodany wiersz z odchyleniem standardowym
        if std_fmt:
            t.append([std_fmt % v for v in stds[:]])
        # Obliczenie wartosci T i P
        T, p = np.array(
            [[ttest_ind(scores[i, :],
                scores[j, :])
            for i in range(len(clfs))]
            for j in range(len(clfs))]
        ).swapaxes(0, 2)
        _ = np.where((p < alpha) * (T > 0))
        conclusions = [list(1 + _[1][_[0] == i])
                    for i in range(n_clfs)]

        t.append([", ".join(["%i" % i for i in c])
                        if len(c) > 0 and len(c) < len(clfs)-1 else ("all" if len(c) == len(clfs)-1 else nc)
                        for c in conclusions])

    # Prezentacja wyników 

        print(tabulate(t, headers))

    # Zapisanie wyników w formacie .tex
        with open('Statistic.txt', 'w') as f:
            f.write(tabulate(t, headers, tablefmt='latex'))
