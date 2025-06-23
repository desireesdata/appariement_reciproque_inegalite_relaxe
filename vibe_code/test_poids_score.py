### Merci Chat GPT ;)

import numpy as np
import random
import jellyfish
from scipy.special import softmax
from scipy.stats import entropy

from get_text_from_json import truth, predicted

# ========== Étape 1 : Données d'exemple (à remplacer par tes vraies données) ==========
# truth = ["Bergeon (Benoît-Charles)", "517", "520", "521", "522", "523", "1517", "Aristide", "Chirac"]
# predicted_clean = une version propre avec légères modifications
def generate_clean_predictions(truth):
    return [t if random.random() > 0.2 else t + " " for t in truth]

# predicted_noisy = version aléatoire et bruitée
def generate_noisy_predictions(truth):
    corrupted = random.sample(truth, len(truth))
    return [t[::-1] if random.random() > 0.5 else t + "X" for t in corrupted]

predicted_clean = generate_clean_predictions(truth)
predicted_noisy = generate_noisy_predictions(truth)


# ========== Étape 2 : Fonctions principales ==========
def levenshtein_matrix(A, B):
    return np.array([[jellyfish.levenshtein_distance(a, b) for b in B] for a in A])

def local_entropy(row, alpha=1.0):
    probs = softmax(-row)
    return alpha * entropy(probs)

def compute_mean_distance(D, matches):
    return np.mean([D[i, j] for i, j in matches])

def compute_mean_entropy(D):
    return np.mean([local_entropy(D[i, :]) for i in range(D.shape[0])])

def find_reciprocal_matches(D):
    n, m = D.shape
    matches = []
    for i in range(n):
        j = np.argmin(D[i])
        i_back = np.argmin(D[:, j])
        if i == i_back:
            matches.append((i, j))
    return matches

# ========== Étape 3 : Calcul pour les deux jeux ==========
def process_matching(truth, predicted):
    D = levenshtein_matrix(truth, predicted)
    matches = find_reciprocal_matches(D)
    mean_dist = compute_mean_distance(D, matches)
    mean_entropy = compute_mean_entropy(D)
    return mean_dist, mean_entropy, np.max(D), np.log(len(predicted))

dist_clean, entropy_clean, d_max, e_max = process_matching(truth, predicted_clean)
dist_noise, entropy_noise, _, _ = process_matching(truth, predicted_noisy)

# ========== Étape 4 : Recherche du meilleur couple de poids ==========
def compute_score(dist, entr, d_max, e_max, w_d, w_e):
    nd = dist / d_max if d_max != 0 else 0
    ne = entr / e_max if e_max != 0 else 0
    return w_d * (1 - nd) + w_e * (1 - ne)

best_gap = -np.inf
best_weights = None
print("\nTest des poids (distance vs entropie) :\n")

for w_d in np.linspace(0.1, 0.9, 9):
    w_e = 1 - w_d
    score_clean = compute_score(dist_clean, entropy_clean, d_max, e_max, w_d, w_e)
    score_noise = compute_score(dist_noise, entropy_noise, d_max, e_max, w_d, w_e)
    gap = score_clean - score_noise
    print(f"Poids d = {w_d:.2f} | e = {w_e:.2f} => Score clean = {score_clean:.3f}, bruité = {score_noise:.3f} | Gap = {gap:.3f}")
    if gap > best_gap:
        best_gap = gap
        best_weights = (w_d, w_e)

print("\n✨ Poids optimaux trouvés expérimentalement :")
print(f"   Distance : {best_weights[0]:.2f}  |  Entropie : {best_weights[1]:.2f}")
print(f"   Meilleur écart de score (clean vs bruité) : {best_gap:.4f}")
