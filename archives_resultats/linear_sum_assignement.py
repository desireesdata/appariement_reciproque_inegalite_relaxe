import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
import jellyfish
from get_text_from_json import truth, predicted

def levenshtein_matrix(A, B):
    return np.array([[jellyfish.levenshtein_distance(a, b) for b in B] for a in A])

D = levenshtein_matrix(truth, predicted)
n, m = D.shape

# Entropie locale autour du meilleur match
def entropy_at_best_match(row, window=5):
    j_best = np.argmin(row)
    half = window // 2
    start = max(0, j_best - half)
    end = min(len(row), j_best + half + 1)
    segment = row[start:end]
    probs = softmax(-segment)
    return entropy(probs)

# Calcul des entropies locales pour chaque ligne
local_epsilons = np.array([entropy_at_best_match(D[i, :], window=5) for i in range(n)])

# Si le nombre de lignes et colonnes est différent, on ajuste la matrice
if n > m:
    # Compléter predicted avec des "dummy" à coût max
    padding = np.full((n, n - m), fill_value=D.max() + 1)
    D_padded = np.hstack((D, padding))
elif m > n:
    # Compléter truth avec des "dummy"
    padding = np.full((m - n, m), fill_value=D.max() + 1)
    D_padded = np.vstack((D, padding))
else:
    D_padded = D

# Appariement optimal
row_ind, col_ind = linear_sum_assignment(D_padded)

# Filtrage pour ignorer les appariements sur les "dummy"
filtered_matches = []
mean_distances = []
matched_truth_indices = set()

for i, j in zip(row_ind, col_ind):
    if i < n and j < m:
        filtered_matches.append((i, j))
        matched_truth_indices.add(i)
        mean_distances.append(D[i, j])

# Affichage
print("Appariements optimaux :\n")
for i in range(n):
    if i in matched_truth_indices:
        j = [j_ for i_, j_ in filtered_matches if i_ == i][0]
        print(f"{truth[i]} <-> {predicted[j]} (d = {D[i, j]}, ε = {local_epsilons[i]:.2f})")
    else:
        print(f"{truth[i]} <aucun match>")

# Statistiques
max_entropy = np.log(m)
mean_entropy = sum(local_epsilons[i] for i in matched_truth_indices) / len(filtered_matches) if filtered_matches else 0
mean_distance = sum(mean_distances) / len(mean_distances) if mean_distances else 0

print(f"\nEntropie moyenne : {mean_entropy:.2f}")
print(f"Distance moyenne : {mean_distance:.2f}")
print(f"Entropie maximale : {max_entropy:.2f}")

# Score de matching
max_len_truth = max(len(s) for s in truth)
max_len_predicted = max(len(s) for s in predicted)
d_max = max(max_len_truth, max_len_predicted)

normalized_distance = mean_distance / d_max
normalized_entropy = mean_entropy / max_entropy

w_d = 0.5
w_e = 0.5
match_score = w_d * (1 - normalized_distance) + w_e * (1 - normalized_entropy ** 0.5)
match_score_percent = match_score * 100

print(f"\nScore de confiance du matching (global) : {match_score:.3f} / {match_score_percent:.1f} %")
coverage = len(set(j for _, j in filtered_matches)) / m
print(f"Taux de couverture des 'predicted' : {coverage:.2%}")
