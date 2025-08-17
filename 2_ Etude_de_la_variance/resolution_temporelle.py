import numpy as np
import matplotlib.pyplot as plt

# Constantes physiques
e = 1.602e-19     # charge élémentaire [C]
me = 9.109e-31    # masse de l'électron [kg]
E0 = 2.3          # énergie de liaison moyenne [eV]
d = 15e-3         # distance entre les plans [m]
V = 100           # tension entre les plans [V]
Er = 1 / d        # champ de Ramo constant entre les plans [V/m]

# Paramètres simulation
N = 1000                         # nombre d'électrons
dt = 1e-12                        # pas de temps [s] (résolution temporelle)
E_field = V / d                 # champ électrique [V/m]
a = e * E_field / me            # accélération en z [m/s²]
z_end = d

# Temps maximum : électron sans vitesse initiale
t_max = np.sqrt(2 * d / a)
n_steps = int(t_max / dt) + 1
time = np.linspace(0, t_max, n_steps)

# Stockage du courant total à chaque instant
Itotal = np.zeros(n_steps)

for i in range(N):
    # --- Génération de l'énergie aléatoire selon N(E) = 1/(E + E0)
    u = np.random.rand()
    E = E0 * (u / (1 - u))   # tirage inverse
    E_J = E * e              # conversion en joules

    # Vitesse initiale totale
    v0 = np.sqrt(2 * E_J / me)

    # Tirage aléatoire dans la demi-sphère supérieure
    theta = np.arccos(np.random.rand())  # angle par rapport à z [0, π/2]
    phi = 2 * np.pi * np.random.rand()   # angle azimutal

    # Composantes de vitesse initiale
    vz0 = v0 * np.cos(theta)

    # Évolution dans le champ électrique constant
    z = vz0 * time + 0.5 * a * time**2
    vz = vz0 + a * time

    # Arrêt de l'électron une fois arrivé à l'électrode opposée
    mask = z >= z_end
    vz[mask] = 0

    # Calcul du courant de Ramo
    I = e * vz / d
    I[mask] = 0

    # Ajout au courant total
    Itotal += I

# Tracé du courant total
plt.figure(figsize=(8, 4))
plt.plot(time * 1e9, Itotal * 1e12)  # temps en ns, courant en pA
plt.xlabel("Temps (ns)")
plt.ylabel("Courant induit (pA)")
plt.title(f"Courant de Shockley-Ramo induit dans l’électrode à 0 V ({N} électrons)")
plt.grid(True)
plt.tight_layout()
plt.show()