import numpy as np
from scipy.optimize import linear_sum_assignment
import json
from get_text_from_json import truth, predicted
import jellyfish

text_values1 = truth
text_values2 = predicted

def levenshtein_matrix(A, B):
    return np.array([[jellyfish.levenshtein_distance(a, b) for b in B] for a in A])

D = levenshtein_matrix(truth, predicted)
n, m = D.shape

similarity_matrix = D
row_ind, col_ind = linear_sum_assignment(similarity_matrix)
matched_in_text_values1 = set(row_ind)

# dico pour les appariements pour un accès rapide
match_dict = {i: j for i, j in zip(row_ind, col_ind)}

# Afficher les résultats dans l'ordre ordinal de text_values1
for i, value in enumerate(text_values1):
    if i in matched_in_text_values1:
        # Si l'élément est apparié, afficher l'appariement
        j = match_dict[i]
        print(f'{value} <-> {text_values2[j]}')
    else:
        # Si l'élément n'est pas apparié, afficher sans correspondance
        print(f'{value} <-> <aucun match>')

# Afficher les éléments de text_values2 qui n'ont pas de correspondance
matched_in_text_values2 = set(col_ind)
for j, value in enumerate(text_values2):
    if j not in matched_in_text_values2:
        print(f'<aucun match> <-> {value}')

# # Afficher les paires de valeurs avec leur distance 
# for i, j in zip(row_ind, col_ind):
#     print(f'"{text_values1[i]}" -----(score : {similarity_matrix[i, j]:.2f})-----> "{text_values2[j]}"')


# # Element non-matchés
# unmatched_in_text_values1 = set(range(len(text_values1))) - set(row_ind)
# unmatched_in_text_values2 = set(range(len(text_values2))) - set(col_ind)

# print("pas de match vt : ")
# for index in unmatched_in_text_values1:
#     print(f"Index: {index}, Value: {text_values1[index]}")

# print("pas de match données evaluéees: ")
# for index in unmatched_in_text_values2:
#     print(f"Index: {index}, Value: {text_values2[index]}")