import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Constantes physiques
# -----------------------------
e = 1.602e-19     # charge élémentaire [C]
me = 9.109e-31    # masse de l'électron [kg]
E0 = 2.3          # énergie de liaison moyenne [eV]
d = 15e-3         # distance entre les plans [m]
V = 100           # tension entre les plans [V]
Er = 1 / d        # champ de Ramo constant entre les plans [V/m]

# -----------------------------
# Paramètres simulation
# -----------------------------
N = 100_000                  # nombre d'électrons
dt = 1e-12                   # pas de temps [s]
E_field = V / d              # champ électrique [V/m]
a = e * E_field / me         # accélération en z [m/s²]
z_end = d

# Temps maximum : électron sans vitesse initiale
t_max = np.sqrt(2 * d / a)
n_steps = int(t_max / dt) + 1
time = np.linspace(0, t_max, n_steps)   # [s]

# -----------------------------
# Simulation (vectorisée par batchs)
# -----------------------------
Itotal = np.zeros(n_steps)
batch_size = 5000
nbatches = (N + batch_size - 1) // batch_size

for b in range(nbatches):
    m = min(batch_size, N - b * batch_size)

    # --- Tirage des énergies selon N(E) = 1/(E+E0)
    u = np.random.rand(m)
    E = E0 * (u / (1 - u))       # [eV]
    E_J = E * e                  # [J]

    # --- Vitesse initiale totale
    v0 = np.sqrt(2 * E_J / me)   # [m/s]

    # --- Direction aléatoire (hémisphère supérieure)
    theta = np.arccos(np.random.rand(m))   # [0, pi/2]
    vz0 = v0 * np.cos(theta)               # [m/s]

    # --- Évolution temporelle
    t = time[None, :]                                   # (1, T)
    z = vz0[:, None] * t + 0.5 * a * t**2               # (m, T)
    vz = vz0[:, None] + a * t                           # (m, T)

    # --- Arrêt à l'électrode opposée
    alive = (z < z_end)                                 # bool (m, T)
    vz = vz * alive

    # --- Courant de Ramo
    I = e * vz / d                                      # (m, T)

    # --- Somme sur les m électrons du batch
    Itotal += I.sum(axis=0)

# -----------------------------
# Recherche du 50% (front descendant) et dérivée locale
# -----------------------------
imax = np.argmax(Itotal)
Imax = Itotal[imax]
Ihalf = 0.5 * Imax

# Premier passage sous Ihalf après le maximum
cross = np.where(Itotal[imax:] <= Ihalf)[0]
if len(cross) == 0:
    raise RuntimeError("Le signal ne retombe pas à 50% après le pic.")

i2 = imax + cross[0]
i1 = max(imax, i2 - 1)  # on reste sur la descente

# Interpolation linéaire pour t50 (I(t50) = Ihalf)
t1, t2 = time[i1], time[i2]
I1, I2 = Itotal[i1], Itotal[i2]
if I2 == I1:
    t50 = t1
else:
    t50 = t1 + (Ihalf - I1) * (t2 - t1) / (I2 - I1)

# Dérivée locale dI/dt autour de t50 par régression linéaire (fenêtre ±5 points)
win = 5
k = np.clip(np.searchsorted(time, t50), 1, len(time) - 2)
i_start = max(0, k - win)
i_end = min(len(time), k + win + 1)
tt = time[i_start:i_end]
II = Itotal[i_start:i_end]
A = np.vstack([tt, np.ones_like(tt)]).T
slope, intercept = np.linalg.lstsq(A, II, rcond=None)[0]  # slope = dI/dt [A/s]

# -----------------------------
# Règle de trois : dt = dA * (1 / |dI/dt|)
# -----------------------------
fractions = [0.05, 0.02, 0.005, 0.001]  # 5%, 2%, 0.5%, 0.1%
abs_slope = abs(slope)

print(f"Imax            = {Imax:.6e} A")
print(f"t50%            = {t50*1e9:.3f} ns")
print(f"(dI/dt)|50%     = {slope:.6e} A/s")
print(f"s_norm = |dI/dt|/Imax = {abs_slope/Imax:.6e} 1/s\n")

print("Résultats (règle de trois  dt = dA * 1/|dI/dt| ) :")
print(" frac(=ΔA/Imax)     ΔA [A]            dt [ps]")
for frac in fractions:
    dA = frac * Imax
    dt_est = dA / abs_slope
    print(f" {frac:>8.3%}   {dA: .6e}    {dt_est*1e12:8.2f}")

# -----------------------------
# Tracé : signal + tangente en t50 (sans axe vertical)
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(time * 1e9, Itotal * 1e12, label="Courant total")

# Marqueur du point à 50% pour repère
plt.scatter([t50 * 1e9], [Ihalf * 1e12], zorder=5, label="Point 50%")

# Tangente en (t50, Ihalf)
t_span = 0.2 * (time[-1] - time[0])  # largeur visuelle de la droite
t_tan = np.linspace(max(time[0], t50 - t_span/2),
                    min(time[-1], t50 + t_span/2), 200)
I_tan = Ihalf + slope * (t_tan - t50)

# Légende de la tangente avec dérivée à 3 chiffres significatifs en A/s
plt.plot(
    t_tan * 1e9, I_tan * 1e12,
    linestyle=":", linewidth=2,
    label=fr"Tangente (dI/dt = {slope:.3e} A/s)"
)

plt.xlabel("Temps (ns)")
plt.ylabel("Courant induit (pA)")
plt.title(f"Courant de Shockley–Ramo (N={N}) et dérivée à 50%")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
