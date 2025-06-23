La seule matrice de similarité qui est un espace métrique est celle qui compare la vérité terrain --> vérité terrain.
Autrement dit, seuls les matchs parfaits respectent les propriétés d'un espace métrique (symétrie / inégalité triangulaire / séparation).

> Rapidement : 1) on exclut toutes les matrices rectangulaires car la symétrie ne peut pas être respectée ("la diagonale tombe toujours à côté d'un des deux coins"). Il n'y a que les matrices carrées $n.n$. La notion de distance exclut toutes les matrices carrés qui ne respectent pas l'inégalité triangulaire parce qu'il est impossible d'avoir $d(AC) > d(AB) + d(BC)$ et ne reste que les matrices vt --> vt qui la respecte.

Cela veut dire que trouver des matchs revient à trouver les zones qui vérifient ces propriétés métriques, mais à condition de relacher l'inégalité triangulaire à cause du bruit.

Le défaut de cette approche est de devoir trouver un seuil de "relâchement" de l'inégalité triangulaire qui fait "émerger" des espaces métriques localement, au sein d'une matrice de similarité. Mais on peut quantifier la relaxation et attribuer un score de confiance aux matches trouvés.

Ici, c'est un calcul d'entropie locale qui a été choisie, mais on pourrait imaginer d'autres méthodes !

Extrait sortie :

```
Babin-Chevaye  <->  Babin-Chevaye  (d = 0, ε = 0.01)
8  <->  8  (d = 0, ε = 4.52)
582  <->  582  (d = 0, ε = 4.59)
719  <->  719  (d = 0, ε = 4.59)
Bachelet Alexandre  <->  Bachelet (Alexandre)  (d = 2, ε = 0.00)
V. Alexandre Bachelet  <->  Alexandre Bachelet  (d = 3, ε = 0.00)
Barthou (Louis)  <->  Barthou (Louis)  (d = 0, ε = 0.01)
2  <->  2  (d = 0, ε = 4.54)
[...]
Borrel (Antoine)  <->  Borrel (Antoine)  (d = 0, ε = 0.00)
1418  <->  1418  (d = 0, ε = 4.55)
Bosc (Jean)  <->  Bosc (Jean)  (d = 0, ε = 0.06)
V. Jean Bosc  <->  Jean Bosc  (d = 3, ε = 0.21)
Bourdeaux (Henry)  <->  Bourdeaux (Henry)  (d = 0, ε = 0.01)
799  <->  799  (d = 0, ε = 4.53)

Entropie maximale : 5.147494476813453
Entropie moyenne : 4.687103415570865 
Distance moyenne : 0.048484848484848485

``` 
Les numéros de pages (listes de nombres) augmentent significativement l'entropie étant donné les faibles distances de Levenshtein.
Cependant leur distance est toujours très faible ou quasi-nulle !

Avec une entropie basée sur une fenêtre glissante (fonction `def entropy_at_best_match()`), on a des résultats plus sûrs :

```
Entropie moyenne : 0.79
Distance moyenne : 0.05
Entropie maximale : 5.15

Score de confiance du matching : 0.803 (sur 1) / 80.3 %
```

Calcul du score :

```
# Score final
w_d = 0.5  # poids de la distance
w_e = 0.5  # poids de l'entropie
match_score = w_d * (1 - normalized_distance) + w_e * (1 - normalized_entropy ** 0.5)
# match_score = (1 - normalized_distance) * (1 - normalized_entropy ** 0.5)
match_score_percent = match_score * 100

print(f"\nScore de confiance du matching : {match_score:.3f} (sur 1) / {match_score_percent:.1f} %")
```