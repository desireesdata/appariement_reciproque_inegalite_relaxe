import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
import jellyfish
from get_text_from_json import truth, predicted

def levenshtein_matrix(A, B):
    return np.array([[jellyfish.levenshtein_distance(a, b) for b in B] for a in A])

# D est une matrice de similarité qui compare truth et predicted
D = levenshtein_matrix(truth, predicted)
n, m = D.shape

# Calcul d'entropie de façon bourrin
def local_entropy(row, alpha=1.0):
    probs = softmax(-row)
    return alpha * entropy(probs)

# Entropie avec fenetre glissante autour de la meilleure val
def entropy_at_best_match(row, window=5):
    j_best = np.argmin(row)
    half = window // 2
    start = max(0, j_best - half)
    end = min(len(row), j_best + half + 1)
    segment = row[start:end]
    probs = softmax(-segment)
    return entropy(probs)

# Relaxation basée sur le gradient local (à voir)
def local_gradient_epsilon(row, beta=1.0):
    grad = np.abs(np.gradient(row))
    score = 1 / (1 + np.mean(grad))
    return beta * score

local_epsilons = np.array([entropy_at_best_match(D[i, :], window=5) for i in range(n)])

# Appariements
matches = []
for i in range(n):
    # On parcourt "la ligne" des VT et j capture l'élément predicted le plus petit (le + proche) de truth
    j = np.argmin(D[i])
    
    # Implémentation du critère de séparation : si la distance est nulle c'est clairement le même objet
    if D[i, j] == 0:
        matches.append((i, j))
        continue

    # Sinon test de symétrie avec :
    # Le parcourt sur la colonne dans laquelle a pied l'élement précédement choisi pour capturer avec i_back le + proche predicted
    i_back = np.argmin(D[:, j]) 
    # Si les deux se "préfèrent" de façon réciproque, alors on ajoute l'ément (D_i,j)
    if i == i_back:
        matches.append((i, j))

# Filtrage par inégalité triangulaire
filtered_matches = []
matched_truth_indices = set()
for i, j in matches:

    # Implémentation du critère de séparation
    if D[i, j] == 0:
        filtered_matches.append((i, j))
        matched_truth_indices.add(i)
        continue

    # Préparation du critère de relaxation
    epsilon = local_epsilons[i]
    is_consistent = True
    
    # On boucle sur toutes les predicted (on ignore les éléments qui vérifie le principe de séparation) et on vérifie l'inégalité triangulaire. Si c'est consistant on ajoute, sinon on rejette !
    for k in range(m):
        if k != j:
            if D[i, j] > D[i, k] + D[np.argmin(D[:, j]), j] + epsilon:
                is_consistent = False
                break
    if is_consistent:
        filtered_matches.append((i, j))
        matched_truth_indices.add(i)

# Affichage complet
print("Appariements (ou absence) :\n")
mean_distances = []
for i in range(n):
    if i in matched_truth_indices:
        j = [j_ for i_, j_ in filtered_matches if i_ == i][0]
        print(f"{truth[i]} <-> {predicted[j]} (d = {D[i, j]}, ε = {local_epsilons[i]:.2f})")
        mean_distances.append(D[i, j])
    else:
        print(f"{truth[i]} <aucun match>")

# Statistiques
max_entropy = np.log(m)
mean_entropy = sum(local_epsilons[i] for i in matched_truth_indices) / len(filtered_matches) if filtered_matches else 0
mean_distance = sum(mean_distances) / len(mean_distances) if mean_distances else 0

print(f"\nEntropie moyenne : {mean_entropy:.2f}")
print(f"Distance moyenne : {mean_distance:.2f}")
print(f"Entropie maximale : {max_entropy:.2f}")

# Distance maximale possible (approche simple)
max_len_truth = max(len(s) for s in truth)
max_len_predicted = max(len(s) for s in predicted)
d_max = max(max_len_truth, max_len_predicted)

# Normalisations
normalized_distance = mean_distance / d_max
normalized_entropy = mean_entropy / max_entropy

# Score final
w_d = 0.5  # poids de la distance
w_e = 0.5  # poids de l'entropie
match_score = w_d * (1 - normalized_distance) + w_e * (1 - normalized_entropy ** 0.5)
# match_score = (1 - normalized_distance) * (1 - normalized_entropy ** 0.5)
match_score_percent = match_score * 100

print(f"\nScore de confiance du matching : {match_score:.3f} (sur 1) / {match_score_percent:.1f} %")
coverage = len(set(j for _, j in filtered_matches)) / m
print(f"Taux de couverture des 'predicted' : {coverage:.2%}")

