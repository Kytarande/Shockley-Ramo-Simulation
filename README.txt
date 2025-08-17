Dernière mise à jour : 2025-08-17 16:12

Description
-----------
Ce projet regroupe des outils numériques pour simuler le fonctionnement d’un détecteur basé sur le courant de Shockley–Ramo. 
Il comprend le calcul des champs électriques et des champs de Ramo, la simulation de trajectoires d’électrons secondaires, 
et le calcul des courants induits dans les électrodes. Des modules additionnels permettent d’étudier la distribution 
des temps de vol et la résolution temporelle.

Scripts
-------
Répertoire : 1_Simulation_courrant_de_Ramo
- 1_champ_avec_interpolation.py : résolution du potentiel et du champ électrique dans un domaine avec électrodes, 
  génération des champs de Ramo.
- 2_trajectoir.py : simulation d’une trajectoire d’électron à partir du champ calculé.
- 3_courant.py : calcul du courant induit dans chaque électrode via le théorème de Shockley–Ramo.
Répertoire : 2_ Etude_de_la_variance
- histogramme.py : génération et analyse statistique des temps de vol par méthode Monte Carlo.
- resolution_temporelle.py : estimation de la résolution temporelle selon le nombre d’électrons secondaires.

Dépendances
-----------
- Python ≥ 3.10
- numpy, scipy, matplotlib
- tqdm (optionnel)

Exécution typique
-----------------
1) Calcul des champs : 
   python 1_champ_avec_interpolation.py
2) Simulation d’une trajectoire : 
   python 2_trajectoir.py
3) Calcul du courant induit : 
   python 3_courant.py
4) Analyses statistiques : 
   python histogramme.py
   python resolution_temporelle.py

Contact
-------
Auteur : Alexandre Poirot — M1 Physique Fondamentale / Magistère — Université Paris Cité
Mail : alexandre.poirot.pro@gmail.com