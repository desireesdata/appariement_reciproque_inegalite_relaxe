import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
import jellyfish

from get_text_from_json import truth, predicted

def levenshtein_matrix(A, B):
    return np.array([[jellyfish.levenshtein_distance(a, b) for b in B] for a in A])

def entropy_at_best_match(row, window=5):
    j_best = np.argmin(row)
    half = window // 2
    start = max(0, j_best - half)
    end = min(len(row), j_best + half + 1)
    segment = row[start:end]
    probs = softmax(-segment)
    return entropy(probs)

def match_sequences(truth, predicted, strict=True, window=5, w_d=0.5, w_e=0.5):
    D = levenshtein_matrix(truth, predicted)
    n, m = D.shape

    local_epsilons = np.array([entropy_at_best_match(D[i, :], window=window) for i in range(n)])

    used_predicted = set()
    matches = []

    for i in range(n):
        j = np.argmin(D[i])
        i_back = np.argmin(D[:, j])

        if D[i, j] == 0:
            if strict and j in used_predicted:
                continue
            matches.append((i, j))
            if strict:
                used_predicted.add(j)
            continue

        if i == i_back:
            if strict and j in used_predicted:
                continue
            matches.append((i, j))
            if strict:
                used_predicted.add(j)

    # Filtrage avec inegalité triangulaire
    filtered_matches = []
    matched_truth_indices = set()
    matched_predicted_indices = set()

    for i, j in matches:
        if D[i, j] == 0:
            filtered_matches.append((i, j))
            matched_truth_indices.add(i)
            matched_predicted_indices.add(j)
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
            matched_predicted_indices.add(j)

    # Affichage des matches
    print("Appariements :")
    mean_distances = []
    for i in range(n):
        match = next(((i_, j_) for i_, j_ in filtered_matches if i_ == i), None)
        if match:
            j = match[1]
            print(f"{truth[i]} <-> {predicted[j]} (d = {D[i, j]}, ε = {local_epsilons[i]:.2f})")
            mean_distances.append(D[i, j])
        else:
            print(f"{truth[i]} <aucun match>")

    # Statistiques
    max_entropy = np.log(m)
    mean_entropy = np.mean([local_epsilons[i] for i in matched_truth_indices]) if matched_truth_indices else 0
    mean_distance = np.mean(mean_distances) if mean_distances else 0
    max_len_truth = max(len(s) for s in truth)
    max_len_predicted = max(len(s) for s in predicted)
    d_max = max(max_len_truth, max_len_predicted)

    normalized_distance = mean_distance / d_max
    normalized_entropy = mean_entropy / max_entropy if max_entropy > 0 else 0

    match_score = w_d * (1 - normalized_distance) + w_e * (1 - normalized_entropy ** 0.5)
    match_score_percent = match_score * 100

    print(f"\nEntropie moyenne : {mean_entropy:.2f}")
    print(f"Distance moyenne : {mean_distance:.2f}")
    print(f"Score de confiance du matching : {match_score:.3f} / {match_score_percent:.1f} %")

    coverage_predicted = len(set(j for _, j in filtered_matches)) / m
    print(f"Taux de couverture des 'predicted' : {coverage_predicted:.2%}")

    return filtered_matches


matches = match_sequences(truth, predicted, strict=True)