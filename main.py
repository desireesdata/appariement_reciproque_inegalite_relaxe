import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
import jellyfish
from get_text_from_json import truth, predicted

"""
La seule matrice de similarité qui est un espace métrique est celle qui compare la vérité terrain --> vérité terrain.
Autrement dit, seuls les matchs parfaits respectent les propriétés d'un espace métrique (symétrie / inégalité triangulaire / séparation).

Cela veut dire que trouver des matchs revient à trouver les zones qui vérifient ces propriétés métriques, mais à condition de relacher l'inégalité triangulaire à cause du bruit.

Le défaut de cette approche est de devoir trouver un seuil de "relâchement" de l'inégalité triangulaire qui fait "émerger" des espaces métriques localement, au sein d'une matrice de similarité. Mais on peut quantifier la relaxation et attribuer un score de confiance aux matches trouvés.

Ici, c'est un calcul d'entropie locale qui a été choisie pour adapter le relâchement, mais on pourrait imaginer d'autres méthodes !
"""

# Matrice de distance Levenshtein
def levenshtein_matrix(A, B):
    return np.array([[jellyfish.levenshtein_distance(a, b) for b in B] for a in A])

# Notre Matrice D est une matrice de similarité qui compare truth et predicted 
D = levenshtein_matrix(truth, predicted)
n, m = D.shape

# Entropie globale (pour quantifier le bruit et ajuster le "relâchement de bride" de l'inégalité triangulaire localement
def local_entropy(row, alpha=1.0):
    probs = softmax(-row)
    return alpha * entropy(probs)

# Entropie avec fenetre glissante autour de la meilleure val
def entropy_at_best_match(row, window=5):
    j_best = np.argmin(row)
    start = max(0, j_best - window // 2)
    end = min(len(row), start + window)
    segment = row[start:end]
    probs = softmax(-segment)
    return entropy(probs)

# Relaxation basée sur le gradient local (à voir)
def local_gradient_epsilon(row, beta=1.0):
    grad = np.abs(np.gradient(row))
    score = 1 / (1 + np.mean(grad))
    return beta * score

# local_epsilons = np.array([local_entropy(D[i, :]) for i in range(n)])
# local_epsilons = [local_gradient_epsilon(D[i, :], beta=1.5) for i in range(n)]
local_epsilons = np.array([entropy_at_best_match(D[i, :], window=5) for i in range(n)])

# Appariements mutuels, qui satisfont la propriété de symétrie
matches = []
for i in range(n):
    j = np.argmin(D[i])         # On parcourt "la ligne" des VT et j capture l'élément predicted le plus petit (le + proche) de truth
    i_back = np.argmin(D[:, j]) # On parcourt la colonne dans laquelle a pied l'élement précédement choisi pour capturer avec i_back le + proche predicted
    if i == i_back:             # Si les deux se "préfèrent" de façon réciproque, alors on ajoute l'ément (D_i,j)
        matches.append((i, j))
    
# Relaxation de l’inégalité triangulaire
# Pour avoir un match "consistant" (car la symétrie ne suffit pas), il faut vérifier pour chaque paire réciproque la propriété de l'inégalité triangulaire, en la relâchant un peu 
filtered_matches = []
for i, j in matches:
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

# Résultats 
max_entropy = np.log(m)
mean_entropy = sum(local_epsilons)/len(filtered_matches)
# median_entropy = np.median()
mean_distances = []

print("Appariements mutuels et cohérents avec la relaxation triangulaire souple): \n")
for i, j in filtered_matches:
    mean_distances.append(D[i,j])
    print(f"{truth[i]}  <->  {predicted[j]}  (d = {D[i, j]}, ε = {local_epsilons[i]:.2f})")

mean_distances = sum(mean_distances)/len(mean_distances)
print(f"\nEntropie moyenne : {mean_entropy} \nDistance moyenne : {mean_distances}")
print(f"Entropie maximale : {max_entropy}")
