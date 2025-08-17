import numpy as np
import matplotlib.pyplot as plt

# Constantes physiques
me = 9.1e-31  # masse de l'électron (kg)
qe = 1.6e-19  # charge élémentaire (C)

# Paramètres de simulation
N = 1000000    # nombre d'électrons
L = 1.5e-2    # distance entre les plaques (m)
dV = 100      # différence de potentiel (V)

# Calcul des constantes
a = (2*me * L) / (qe * dV)
b = (qe * dV) / (2 * me)

def generate_secondary_energies_inverse(N, E0=2.0, Emax=100.0):
    """Génère N énergies selon la loi N(E) ∝ 1 / (E + E0)^2"""
    u = np.random.uniform(0, 1, N)
    E = E0 * (1 / u - 1)
    return np.clip(E, 0, Emax)

def generate_secondary_energies_exp(N, Ec=5.0, Emax=100.0):
    """Génère N énergies selon la loi N(E) ∝ E * exp(-E/Ec)"""
    energies = []
    Pmax = Ec * np.exp(-1)
    while len(energies) < N:
        E_trial = np.random.uniform(0, Emax)
        if np.random.uniform(0, Pmax) < E_trial * np.exp(-E_trial / Ec):
            energies.append(E_trial)
    return np.array(energies)

def vitesse(E):
    """Calcule la vitesse à partir de l'énergie en eV"""
    return np.sqrt(2 * E * qe / me)

def temps_de_vol(teta, v0):
    """Calcule le temps de vol"""
    v0y = np.cos(teta) * v0
    return a * (np.sqrt(v0y**2 + b) - v0y)

# Génération des énergies
E1 = generate_secondary_energies_inverse(N)
E2 = generate_secondary_energies_exp(N)

# Calcul des vitesses
v1 = vitesse(E1)
v2 = vitesse(E2)

# Angles aléatoires
teta = np.arccos(np.random.rand(N))

# Calcul des temps de vol
temps1 = temps_de_vol(teta, v1)
temps2 = temps_de_vol(teta, v2)

# Statistiques
mean1, std1 = np.mean(temps1), np.std(temps1)
mean2, std2 = np.mean(temps2), np.std(temps2)

# Création des figures
plt.figure(figsize=(14, 6))

# Histogramme pour la première distribution
plt.subplot(1, 2, 1)
counts, bins, _ = plt.hist(temps1 * 1e9, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(mean1 * 1e9, color='red', linestyle='dashed', linewidth=2, 
            label=f'Moyenne: {mean1*1e9:.2f} ns')
plt.axvline(mean1*1e9 - std1*1e9, color='green', linestyle='dashed', linewidth=1)
plt.axvline(mean1*1e9 + std1*1e9, color='green', linestyle='dashed', linewidth=1,
            label=f'Écart-type: ±{std1*1e9:.2f} ns')
plt.fill_betweenx([0, max(counts)], 
                  (mean1-std1)*1e9, (mean1+std1)*1e9, 
                  color='green', alpha=0.1)
plt.title('Loi en 1/(E+E0)²')
plt.xlabel('Temps de vol (ns)')
plt.ylabel('Nombre d\'électrons')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Histogramme pour la seconde distribution
plt.subplot(1, 2, 2)
counts, bins, _ = plt.hist(temps2 * 1e9, bins=50, color='salmon', edgecolor='black', alpha=0.7)
plt.axvline(mean2 * 1e9, color='red', linestyle='dashed', linewidth=2, 
            label=f'Moyenne: {mean2*1e9:.2f} ns')
plt.axvline(mean2*1e9 - std2*1e9, color='purple', linestyle='dashed', linewidth=1)
plt.axvline(mean2*1e9 + std2*1e9, color='purple', linestyle='dashed', linewidth=1,
            label=f'Écart-type: ±{std2*1e9:.2f} ns')
plt.fill_betweenx([0, max(counts)], 
                  (mean2-std2)*1e9, (mean2+std2)*1e9, 
                  color='purple', alpha=0.1)
plt.title('Loi en E·exp(-E/Ec)')
plt.xlabel('Temps de vol (ns)')
plt.ylabel('Nombre d\'électrons')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Affichage des résultats numériques
print("\nLoi N(E) ∝ 1/(E + E0)²")
print(f"Temps moyen de vol: {mean1*1e9:.2f} ns")
print(f"Écart-type: {std1*1e9:.2f} ns")

print("\nLoi N(E) ∝ E·exp(-E/Ec)")
print(f"Temps moyen de vol: {mean2*1e9:.2f} ns")
print(f"Écart-type: {std2*1e9:.2f} ns")