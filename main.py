import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
import jellyfish
from get_text_from_json import truth, predicted

def levenshtein_matrix(A, B):
    return np.array([[jellyfish.levenshtein_distance(a, b) for b in B] for a in A])

D = levenshtein_matrix(truth, predicted)
n, m = D.shape

def entropy_at_best_match(row, window=5):
    j_best = np.argmin(row)
    half = window // 2
    start = max(0, j_best - half)
    end = min(len(row), j_best + half + 1)
    segment = row[start:end]
    probs = softmax(-segment)
    return entropy(probs)

# 1. Entropies locales
local_epsilons = np.array([entropy_at_best_match(D[i, :], window=5) for i in range(n)])

# 2. Appariements parfaits (séparation stricte)
perfect_matches = [(i, j) for i in range(n) for j in range(m) if D[i, j] == 0]
matched_truth = {i for i, _ in perfect_matches}
matched_pred = {j for _, j in perfect_matches}

# 3. Appariements symétriques non parfaits
matches = []
for i in range(n):
    if i in matched_truth:
        continue
    j = np.argmin(D[i])
    if j in matched_pred:
        continue
    i_back = np.argmin(D[:, j])
    if i == i_back:
        matches.append((i, j))

# 4. Fusion des deux
all_matches = perfect_matches + matches

# 5. Filtrage avec inégalité triangulaire
filtered_matches = []
matched_truth_indices = set()

for i, j in all_matches:
    if D[i, j] == 0:
        filtered_matches.append((i, j))
        matched_truth_indices.add(i)
        continue

    epsilon = local_epsilons[i]
    is_consistent = True
    for k in range(m):
        if k != j:
            if D[i, j] > D[i, k] + D[np.argmin(D[:, j]), j] + epsilon:
                is_consistent = False
                break
    if is_consistent:
        filtered_matches.append((i, j))
        matched_truth_indices.add(i)

# 6. Affichage des appariements
# print("Appariements (ou absence) :\n")
mean_distances = []
for i in range(n):
    if i in matched_truth_indices:
        j = [j_ for i_, j_ in filtered_matches if i_ == i][0]
        print(f"{truth[i]} <-> {predicted[j]}")
        # print(f"{truth[i]} <-> {predicted[j]} (d = {D[i, j]}, ε = {local_epsilons[i]:.2f})")
        mean_distances.append(D[i, j])
    else:
        print(f"{truth[i]} <aucun match>")

# 7. Statistiques globales
max_entropy = np.log(m)
mean_entropy = sum(local_epsilons[i] for i in matched_truth_indices) / len(filtered_matches) if filtered_matches else 0
mean_distance = sum(mean_distances) / len(mean_distances) if mean_distances else 0

print(f"\nEntropie moyenne : {mean_entropy:.2f}")
print(f"Distance moyenne : {mean_distance:.2f}")
print(f"Entropie maximale : {max_entropy:.2f}")

# 8. Normalisation + score global
max_len_truth = max(len(s) for s in truth)
max_len_predicted = max(len(s) for s in predicted)
d_max = max(max_len_truth, max_len_predicted)

normalized_distance = mean_distance / d_max
normalized_entropy = mean_entropy / max_entropy

w_d = 0.5
w_e = 0.5
match_score = w_d * (1 - normalized_distance) + w_e * (1 - normalized_entropy ** 0.5)
match_score_percent = match_score * 100

print(f"\nScore de confiance du matching : {match_score:.3f} (sur 1) / {match_score_percent:.1f} %")
coverage = len(set(j for _, j in filtered_matches)) / m
print(f"Taux de couverture des 'predicted' : {coverage:.2%}")
