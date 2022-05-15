import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import rankdata
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from tabulate import tabulate
from scipy.stats import ttest_ind
from RSM import RSM

#metody zespołowe
ensembles = {
    'none': None,
    'bagging': BaggingClassifier(base_estimator=GaussianNB()),
    'RSM': RSM(base_estimator=GaussianNB()),
}

#import wyników
scores= np.load("results.npy")
scores = scores[:,0,:]
#________________________TESTY STATYSTYCZNE_____________________
"""Przeprowadzono testy statystyczne w celu określenia, która metoda preprocesingu jest najlepszana określonym klasyfikatorze"""
alfa = .05
t_statistic = np.zeros((len(ensembles), len(ensembles)))
p_value = np.zeros((len(ensembles), len(ensembles)))

for i in range(len(ensembles)):
    for j in range(len(ensembles)):
        t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])

headers = list(ensembles.keys())
names_column = np.expand_dims(np.array(list(ensembles.keys())), axis=1)
t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\n\n\tStatistic tests for GNB classificator")
print("\n\nGNB t-statistic:\n\n", t_statistic_table, "\n\n GNB p-value:\n\n", p_value_table)

#Tablica przewag
advantage = np.zeros((len(ensembles), len(ensembles)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\n\nAdvantage: \n\n", advantage_table)

#Róznice statystyczne znaczące
significance = np.zeros((len(ensembles), len(ensembles)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
print(f"\n\nStatistical significance (alpha = {alfa} ):\n\n", significance_table)

#Wyniki koncowe analizy statystycznej
stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)
print("\n\nStatistically significantly better:\n\n", stat_better_table)

