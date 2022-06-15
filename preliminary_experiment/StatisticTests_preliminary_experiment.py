from matplotlib.pyplot import axis
import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_ind
from preliminary_experiment import ensembles

#ZMIENNE GLOBALNE
alpha=.05
m_fmt="%.3f"
nc="---"
db_fmt="%s"
tablefmt="plain"
clfs=ensembles
n_clfs = len(clfs)
headers = list(clfs.keys())

if __name__=="__main__":
    #POBRANIE WYNIKÓW
    scores = np.load("results_preliminary_experiment.npy")
    mean_scores = np.mean(scores, axis=1)
    t = []
    
    t.append([m_fmt % v for v in mean_scores[:]])
    #TEST T-STUDEN, WYLICZENIE WARTOSCI T I p
    T, p = np.array(
        [[ttest_ind(scores[i, :], scores[j,:])
        for i in range(len(clfs))]
        for j in range(len(clfs))]
    ).swapaxes(0, 2)
    _ = np.where((p < alpha) * (T > 0))
    conclusions = [list(1 + _[1][_[0] == i])
                for i in range(n_clfs)]
    t.append([", ".join(["%i" % i for i in c])
                    if len(c) > 0 and len(c) < len(clfs)-1 else ("all" if len(c) == len(clfs)-1 else nc)
                    for c in conclusions])

    #WYŚWIETLENIE UZYSKANYCH WYNIKOW
    print(tabulate(t, headers))

    #ZAPISANIE WYNIKÓW DO FORMATU W LATEX
    with open('Statistic_preliminary_experiment.txt', 'w') as f:
        f.write(tabulate(t, headers, tablefmt='latex'))