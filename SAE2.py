# Importation
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
import sys
import heapq
import math

#2.1 Dessin d’un graphe
#Fonction pour dessiner un graphe en utilisant NetworkX et visualiser les plus courts chemins.
def dessiner_graphe(matrice, chemins=None):
    n = len(matrice)
    G = nx.DiGraph()

    # Ajouter les arêtes au graphe
    for i in range(n):
        for j in range(n):
            if matrice[i, j] != float('inf'):
                G.add_edge(i, j, weight=matrice[i, j])

    # Spécifier les positions des sommets pour une meilleure visualisation
    positions = nx.spring_layout(G)

    # Dessiner les sommets et les arêtes
    nx.draw_networkx_nodes(G, positions, node_color='lightblue')
    nx.draw_networkx_edges(G, positions)
    nx.draw_networkx_labels(G, positions)

    # Dessiner les poids des arêtes
    etiquettes_arêtes = {(i, j): matrice[i, j] for i in range(n) for j in range(n) if matrice[i, j] != float('inf')}
    nx.draw_networkx_edge_labels(G, positions, edge_labels=etiquettes_arêtes)

    # les plus courts chemins s'ils sont fournis?
    if chemins:
        for chemin in chemins.values():
            if isinstance(chemin[1], list):
                arêtes_chemin = [(chemin[1][i], chemin[1][i + 1]) for i in range(len(chemin[1]) - 1)]
                nx.draw_networkx_edges(G, positions, edgelist=arêtes_chemin, edge_color='red', width=2.0)

    plt.axis('off')
    plt.show()
    

# 2.2 Dessin d’un chemin
def get_chemin(depart, arrivee, pred):
    """
    Fonction auxiliaire pour récupérer le chemin du sommet de départ au sommet d'arrivée en utilisant le dictionnaire de prédécesseur.

    Args:
        depart: L'indice du sommet de départ.
        arrivee: L'indice du sommet d'arrivée.
        pred: Le dictionnaire de prédécesseur.

    Returns:
        Le chemin sous forme d'une liste d'indices de sommets.
    """
    chemin = [arrivee]
    actuel = arrivee
    while actuel != depart:
        actuel = pred[actuel]
        if actuel is None:
            return 'sommet non joignable à d par un chemin dans le graphe'
        chemin.append(actuel)
    chemin.reverse()
    return chemin



# 3.1 Graphes avec 50% de flèches
def graphe(n, a, b):
    # Génération d'une matrice avec 50% de 0 et 50% de 1
    matrice = np.random.choice([0, 1], size=(n, n), p=[0.5, 0.5])
    
    # Remplacement des 1 par des valeurs aléatoires entre a et b
    random_values = np.random.randint(a, b, size=(n, n))
    matrice[matrice == 1] = random_values[matrice == 1]
    
    # Conversion des 0 en ∞
    matrice = matrice.astype('float64')
    matrice[matrice == 0] = float('inf')
    
    return matrice


#3.2 Graphes avec une proportion variables p de flèches
# Générer une matrice de graphe pondéré aléatoire avec proposition de p
def graphe2(n, p, a, b):
    matrice = np.random.binomial(1, p, size=(n, n))
    valeurs_aléatoires = np.random.randint(a, b, size=(n, n))
    matrice[matrice == 1] = valeurs_aléatoires[matrice == 1]
    matrice = matrice.astype('float64')
    matrice[matrice == 0] = float('inf')
    return matrice

# Exemple d'utilisation
matrice = graphe2(8, 0.2, 1, 10)

print(matrice)
dessiner_graphe(matrice)

# 4.1 Codage de l’algorithme de Dijkstra
def formatResultat(M, d, distances, predecesseurs):
    # Fonction pour formater les résultats de l'algorithme de Dijkstra
    resultats = {}
    for s in range(len(distances)):
        if s == d:
            continue
        if distances[s] == sys.maxsize:
            # Si le sommet n'est pas atteignable, on indique qu'il n'y a pas de chemin
            resultat = (distances[s], "sommet non joignable par un chemin dans le graphe G")
        else:
            # Sinon, on récupère le chemin à partir des prédécesseurs
            chemin = get_chemin(d, s, predecesseurs)
            resultat = (distances[s], chemin)
        resultats[s] = resultat
    return resultats

# Fonction pour savoir si le graphe a des poids négatifs pour Dijkstra
def estPondéréNégatif(M):
    # Parcourt la matrice pour vérifier si des poids sont négatifs
    for i in range(len(M)):
        for j in range(len(M)):
            if M[i][j] < 0:
                return True
    return False



def Dijkstra(M, d):
    # Vérifie si la matrice a des poids négatifs pour l'algorithme de Dijkstra
    assert estPondéréNégatif(M) == False, "Matrice à poids négatif"
    nb_sommets = len(M)
    distances = [sys.maxsize] * nb_sommets
    predecesseurs = [None] * nb_sommets
    visites = [False] * nb_sommets

    distances[d] = 0
    heap = [(0, d)]

    while heap:
        # Sélectionne le sommet avec la plus petite distance actuelle
        distance_actuelle, sommet_courant = heapq.heappop(heap)
        if visites[sommet_courant]:
            continue
        visites[sommet_courant] = True

        for s in range(nb_sommets):
            poids = M[sommet_courant][s]
            if poids > 0 and not visites[s]:
                # Calcule la nouvelle distance à partir du sommet courant
                nouvelle_distance = distance_actuelle + poids
                if nouvelle_distance < distances[s]:
                    # Met à jour la distance et le prédécesseur si la nouvelle distance est plus courte
                    distances[s] = nouvelle_distance
                    predecesseurs[s] = sommet_courant
                    heapq.heappush(heap, (nouvelle_distance, s))

    # Retourne les résultats formatés de l'algorithme de Dijkstra
    return formatResultat(M, d, distances, predecesseurs)


# Exemple
matrice = np.array([
    [0, 1, 3, float('inf'), float('inf')],
    [1, 0, float('inf'), 2, float('inf')],
    [3, float('inf'), 0, 1, 5],
    [float('inf'), 2, 1, 0, 4],
    [float('inf'), float('inf'), 5, 4, 0]
])

sommet_depart = 0
resultat = Dijkstra(matrice, sommet_depart)

print("Plus courts chemins:")
for sommet, (distance, chemin) in resultat.items():
    print(f"De {sommet_depart} à {sommet}:")
    print(f"Distance: {distance}")
    print(f"Chemin: {chemin}")
    print()

dessiner_graphe(matrice, resultat)

#4.2 Codage de l’algorithme de Belman-Ford
def Bellman_Ford(M, d):
    n = len(M)  # Nombre de sommets dans le graphe
    distances = [float('inf')] * n  # Tableau des distances initialement à l'infini
    distances[d] = 0  # Distance de la source à elle-même est de 0
    predecesseurs = [None] * n  # Tableau des prédécesseurs initialement à None

    # Relâchement des arêtes pour mettre à jour les distances
    for i in range(n - 1):  # Répéter (n - 1) fois
        for u in range(n):
            for v in range(n):
                poids_uv = M[u][v]  # Poids de l'arête (u, v)
                if poids_uv is not None:
                    if distances[u] + poids_uv < distances[v]:
                        distances[v] = distances[u] + poids_uv
                        predecesseurs[v] = u

    # Vérification de l'existence d'un cycle de poids négatif
    for u in range(n):
        for v in range(n):
            poids_uv = M[u][v]  # Poids de l'arête (u, v)
            if poids_uv is not None:
                if distances[u] + poids_uv < distances[v]:
                    return "Sommet joignable depuis d par un chemin, mais présence d'un cycle négatif."

    # Construction des itinéraires ou mention de non-joignabilité
    resultats = []
    for s in range(n):
        if s != d:
            if distances[s] == float('inf'):
                resultats.append(f"Sommet {s} non joignable depuis d par un chemin dans le graphe.")
            else:
                chemin = []
                sommet_courant = s
                while sommet_courant is not None:
                    chemin.insert(0, sommet_courant)
                    sommet_courant = predecesseurs[sommet_courant]
                resultats.append((distances[s], chemin))

    return resultats


# Exemple
matrice = graphe2(5, 0.5, 0, 10)
print(matrice)

source = 1
resultats = Bellman_Ford(matrice, source)

path = None
for resultat in resultats:
    print(resultat)
    if isinstance(resultat, str):
        print(resultat)
    else:
        _, path = resultat
        break

dessiner_graphe(matrice, chemins={source: (None, path)} if path is not None else None)




# 5 Influence du choix de la liste ordonn´ee des flèches pourl’algorithme de Bellman-Ford
# parcours en largeur
def pl(M, s):
    n = len(M)
    couleur = {}     # On colorie tous les sommets en blanc et s (départ) en vert
    for i in range(n):
        couleur[i] = 'blanc'
    couleur[s] = 'vert'
    file = [s]
    Resultat = [s]
    while file != []:
        i = file[0]           # on prend le premier terme de la file
        for j in range(n):  # On enfile les successeurs de i encore blancs:
            if M[file[0]][j] == 1 and couleur[j] == 'blanc':
                file.append(j)
                couleur[j] = 'vert'  # On les colorie en vert (sommets visités)
                Resultat.append(j)  # On les place dans la liste Resultat
        file.pop(0)  # on défile i (on retire le premier élément)
    return Resultat

# parcours en profondeur
def pp(M, s):
    n = len(M)       # taille du tableau = nombre de sommets
    couleur = {}     # On colorie tous les sommets en blanc et s en vert
    for i in range(n):
        couleur[i] = 'blanc'
    couleur[s] = 'vert'
    pile = [s]       # on initialise la pile à s
    Resultat = [s]  # on initialise la liste des résultats à s

    while pile != []:  # tant que la pile n'est pas vide,
        i = pile[-1]          # on prend le dernier sommet i de la pile
        Succ_blanc = []       # on crée la liste de ses successeurs non déjà visités (blancs)
        for j in range(n):
            if M[i, j] == 1 and couleur[j] == 'blanc':
                Succ_blanc.append(j)
        if Succ_blanc != []:  # s'il y en a,
            v = Succ_blanc[0]    # on prend le premier (si on veut l'ordre alphabétique)
            couleur[v] = 'vert'   # on le colorie en vert,
            pile.append(v)      # on l'empile
            Resultat.append(v)  # on le met en liste rsultat
        else:               # sinon:
            pile.pop()          # on sort i de la pile

    return Resultat


# Bellman-Ford en 3 variantes selon le mode de construction dela liste des flèches.
def Bellman_Ford(M, d, ordre):
    n = len(M)  # Nombre de sommets dans le graphe
    distances = [float('inf')] * n  # Tableau des distances initialement à l'infini
    distances[d] = 0  # Distance de la source à elle-même est de 0
    predecesseurs = [None] * n  # Tableau des prédécesseurs initialement à None

    if ordre == 'aleatoire':
        fleches = []
        for u in range(n):
            for v in range(n):
                poids_uv = M[u][v]  # Poids de l'arête (u, v)
                if poids_uv is not None:
                    fleches.append((u, v))
        random.shuffle(fleches)
    elif ordre == 'largeur':
        fleches = []
        parcours = pl(M, d)
        for u in parcours:
            for v in range(n):
                poids_uv = M[u][v]  # Poids de l'arête (u, v)
                if poids_uv is not None:
                    fleches.append((u, v))
    elif ordre == 'profondeur':
        fleches = []
        parcours = pp(M, d)
        for u in parcours:
            for v in range(n):
                poids_uv = M[u][v]  # Poids de l'arête (u, v)
                if poids_uv is not None:
                    fleches.append((u, v))

    # Relâchement des arêtes pour mettre à jour les distances
    tours = 0
    for _ in range(n - 1):  # Répéter (n - 1) fois
        modifie = False
        for u, v in fleches:
            poids_uv = M[u][v]  # Poids de l'arête (u, v)
            if distances[u] + poids_uv < distances[v]:
                distances[v] = distances[u] + poids_uv
                predecesseurs[v] = u
                modifie = True
        if not modifie:
            break
        tours += 1

    # Vérification de l'existence d'un cycle de poids négatif
    for u, v in fleches:
        poids_uv = M[u][v]  # Poids de l'arête (u, v)
        if distances[u] + poids_uv < distances[v]:
            return "Sommet joignable depuis d par un chemin, mais présence d'un cycle négatif."

    # Construction des itinéraires
    resultats = []
    for s in range(n):
        if s != d:
            if distances[s] == float('inf'):
                resultats.append(f"Sommet {s} non joignable depuis d par un chemin dans le graphe.")
            else:
                chemin = []
                sommet_courant = s
                while sommet_courant is not None:
                    chemin.insert(0, sommet_courant)
                    sommet_courant = predecesseurs[sommet_courant]
                resultats.append((distances[s], chemin))

    return resultats, tours

# Génération d'une matrice de graphe pondéré aléatoire de taille n
n = 50
matrice = np.random.randint(1, 10, size=(n, n)).astype('float64')

# fonction Bellman-Ford
resultats_aleatoire, tours_aleatoire = Bellman_Ford(matrice, 0, 'aleatoire')
resultats_largeur, tours_largeur = Bellman_Ford(matrice, 0, 'largeur')
resultats_profondeur, tours_profondeur = Bellman_Ford(matrice, 0, 'profondeur')

# Affichage des résultats 
print("Résultats (ordre aléatoire) :", resultats_aleatoire)
print("Nombre de tours (ordre aléatoire) :", tours_aleatoire)
print()
print("Résultats (ordre largeur) :", resultats_largeur)
print("Nombre de tours (ordre largeur) :", tours_largeur)
print()
print("Résultats (ordre profondeur) :", resultats_profondeur)
print("Nombre de tours (ordre profondeur) :", tours_profondeur)



# fonction pour calculer le temps pour faire algorithme de Dijkstra qui prend en entrée un entier positif n(Taille de matrice)
def TempsDij(n):
    # Générer une matrice aléatoire de taille n 
    matrice = np.random.randint(1, 10, size=(n, n)).astype('float64')

    # Initialiser 
    debut = time.time()

    # Algorithme de Dijkstra 
    Dijkstra(matrice, 0)

    temps_calcul = time.time() - debut

    return temps_calcul

temps = TempsDij(1000)
print("Temps de calcul TempsDij :", temps, "secondes")

def TempsBF(n):
    # Génération d'une matrice aléatoire de taille n
    matrice = np.random.randint(1, 10, size=(n, n)).astype('float64')

    d = 0  # Sommet de départ

    # l'ordre aléatoire
    debut = time.time()
    Bellman_Ford(matrice, d, 'aleatoire')
    temps_aleatoire = time.time() - debut

    # l'ordre en largeur
    debut = time.time()
    Bellman_Ford(matrice, d, 'largeur')
    temps_largeur = time.time() - debut

    # l'ordre en profondeur
    debut = time.time()
    Bellman_Ford(matrice, d, 'profondeur')
    temps_profondeur = time.time() - debut

    return temps_aleatoire, temps_largeur, temps_profondeur

n = 50
temps_aleatoire, temps_largeur, temps_profondeur = TempsBF(n)

print("Temps de calcul (ordre aléatoire) :", temps_aleatoire)
print("Temps de calcul (ordre largeur) :", temps_largeur)
print("Temps de calcul (ordre profondeur) :", temps_profondeur)


temps_aleatoire, temps_largeur, temps_profondeur = TempsBF(200)
print("l'exposant a est", (math.log(temps_aleatoire) - math.log(TempsBF(100)[0])) / (math.log(200) - math.log(100)))

#Fonctions qui calcule le temps d'éxécution de Bellman-Ford pour une taille n donnée avec une proportion qui diminue en fonction de n
def tempsBF_alternatif(n):
    M = graphe2(n, 1/n, 1, 10) # On crée une matrice avec une proportion de 1/n
    start = time.perf_counter() # On initialise le temps avec une variable start
    Bellman_Ford(M, 0, "aleatoire") # On lance 
    stop = time.perf_counter() # On garde dans une seconde variable le temps de fin
    return stop-start # On retourne la différence des deux variables pour avoir le temps d'éxécution

#exemple
print("Temps alternatif:", tempsBF_alternatif(50))

# 7 Test de forte connexité 
# on a utilisé Trans2(M) de R2.07 pour créer un matrice forte connexe
def Trans2(M):
    k = len(M)
    for s in range(k):
        for r in range(k):
            if M[r, s] == 1:
                for t in range(k):
                    if M[s, t] == 1:
                        M[r, t] = 1
    return M
def fc(M):
    # Calcul de la fermeture transitive de la matrice
    T = Trans2(M)

    # Vérification de la forte connexité
    k = len(M)
    for i in range(k):
        for j in range(k):
            if T[i, j] != 1:
                # Si un chemin n'existe pas entre les sommets i et j, le graphe n'est pas fortement connexe
                return False

    return True

# Exemple 
M = np.array([[0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1],
              [0, 0, 1, 0, 0],
              [0, 1, 0, 0, 0]])

is_connexe = fc(M)
print(is_connexe)

#8 Forte connexitè pour un graphe avec p=50% de flèches
def test_stat_fc(n):
    nb_tests = 100  # Nombre de tests à effectuer
    nb_fortement_connexes = 0  # Nombre de graphes fortement connexes

    for i in range(nb_tests):
        M = np.random.binomial(1, 0.5, size=(n, n))  # Génération d'une matrice aléatoire avec des valeurs entre 1 et 10
        if fc(M):
            nb_fortement_connexes += 1

    pourcentage_fortement_connexes = (nb_fortement_connexes / nb_tests) * 100

    return pourcentage_fortement_connexes

# Exemple 
n = 50
pourcentage = test_stat_fc(n)
print(f"Pourcentage de graphes avec p=50% de flèches fortement connexes pour n={n} : {pourcentage}%")



#9 Détermination du seuil de forte connexité
def test_stat_fc2(n, p):
    nb_graphes_fortement_connexes = 0
    nb_tests = 100  # Nombre de tests à effectuer
    
    for i in range(nb_tests):
        matrice = np.random.binomial(1, p, size=(n, n))  # Génération d'une matrice de graphe avec des valeurs aléatoires entre 1 et 10
        
        if fc(matrice):
            nb_graphes_fortement_connexes += 1
    
    pourcentage_fortement_connexes = (nb_graphes_fortement_connexes / nb_tests) * 100
    return  pourcentage_fortement_connexes

# Exemple d'utilisation
n = 50
p = 0.5
pourcentage = test_stat_fc2(n, p)
print(f"Pourcentage de graphes fortement connexes pour n={n}, p={p} : {pourcentage}%")

for i in range(2,20):
    print("pour matrice de taille ", i , "-->", test_stat_fc2(i,0.5))

def seuil(n):
    p = 0.5  
    while test_stat_fc2(n, p) >= 0.99:
        p -=1/100
    return p

#10.1 Représentation graphique de seuil(n)
n_values = range(10, 40)
seuil_values = [seuil(n) for n in n_values]

plt.scatter(n_values, seuil_values)
plt.xlabel('Taille du graphe (n)')
plt.ylabel('Seuil de forte connexité (p)')
plt.title('Représentation graphique de seuil(n)')
plt.show()

# Régression linéaire sur une échelle logarithmique
log_n_values = np.log(n_values)
log_seuil_values = np.log(seuil_values)
coefficients = np.polyfit(log_n_values, log_seuil_values, 1)
a, c = coefficients[0], np.exp(coefficients[1])

# Représentation graphique
plt.scatter(log_n_values, log_seuil_values)
plt.plot(log_n_values, np.polyval(coefficients, log_n_values), color='red', label='Régression linéaire')
plt.xlabel('log(Taille du graphe)')
plt.ylabel('log(Seuil de forte connexité)')
plt.title('Régression linéaire sur le graphe log-log')
plt.legend()
plt.show()

print(f"La fonction seuil(n) est approximativement de la forme p = {c} * n^{a}")
