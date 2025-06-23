La seule matrice de similarité qui est un espace métrique est celle qui compare la vérité terrain --> vérité terrain.
Autrement dit, seuls les matchs parfaits respectent les propriétés d'un espace métrique (symétrie / inégalité triangulaire / séparation).

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
``` 
