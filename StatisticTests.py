from matplotlib.pyplot import axis
import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_ind
from main import ensembles, datasets

# Zmienne globalne uzyte w testach statystycznych
alpha=.05
m_fmt="%.3f"
std_fmt=0
nc="---"
db_fmt="%s"
tablefmt="plain"
clfs=ensembles
n_clfs = len(clfs)
headers = list(clfs.keys())

if __name__=="__main__":
    # Pobranie wyników
    scores = np.load("results.npy")
    mean_scores = np.mean(scores, axis=2)
    stds = np.std(scores, axis=2)
    t = []

    for db_idx, db_name in enumerate(datasets):
        # Wiersz z wartoscia srednia
        t.append([db_fmt % db_name] + [m_fmt % v for v in mean_scores[:, db_idx]])
        # Jesli podamy std_fmt w zmiennych globalnych zostanie do tabeli dodany wiersz z odchyleniem standardowym
        if std_fmt:
            t.append(['']+[std_fmt % v for v in stds[:, db_idx]])
        # Obliczenie wartosci T i P
        T, p = np.array(
            [[ttest_ind(scores[i, db_idx, :],
                scores[j, db_idx, :])
            for i in range(len(clfs))]
            for j in range(len(clfs))]
        ).swapaxes(0, 2)
        _ = np.where((p < alpha) * (T > 0))
        conclusions = [list(1 + _[1][_[0] == i])
                    for i in range(n_clfs)]

        t.append(['']+[", ".join(["%i" % i for i in c])
                        if len(c) > 0 and len(c) < len(clfs)-1 else ("all" if len(c) == len(clfs)-1 else nc)
                        for c in conclusions])

# Prezentacja wyników 
print(tabulate(t, headers))

# Zapisanie wyników w formacie .tex
with open('Statistic.txt', 'w') as f:
    f.write(tabulate(t, headers, tablefmt='latex'))
