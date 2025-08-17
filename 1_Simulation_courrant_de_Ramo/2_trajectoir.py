import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# --- Constantes physiques ---
CHARGE_ELECTRON = -1.602e-19  # Charge de l'électron en Coulombs (C)
MASSE_ELECTRON = 9.109e-31    # Masse de l'électron en kilogrammes (kg)

# --- Fonction d'entrée sécurisée avec valeur par défaut ---
def input_default(prompt, default, type_=float):
    val = input(f"{prompt} [{default}] : ").strip()
    return type_(val) if val else default

print("=== Simulation de trajectoire d'électron ===")

# --- 1. Charger les données de champ électrique ---
try:
    data = np.load("champ_electrique_donnees_ROI.npz", allow_pickle=True)
except FileNotFoundError:
    print("Erreur : Le fichier 'champ_electrique_donnees_ROI.npz' est introuvable.")
    print("Veuillez d'abord exécuter le script de simulation du champ électrique.")
    exit()

# Récupération des données du champ principal (interpolées et filtrées)
X_interp_flat = data['X_interp']
Y_interp_flat = data['Y_interp']
Ex_interp_flat = data['Ex_interp']
Ey_interp_flat = data['Ey_interp']
h_interp = data['h_interp']
nb_bandes = data['nb_bandes']
band_width = data['band_width']
gap = data['gap']
total_width_bottom = data['total_width_bottom']
plan_gap = data['plan_gap']
width_100 = data['width_100']


print("\nDonnées de champ électrique chargées avec succès.")
print(f"Discrétisation de l'interpolation : {h_interp:.2f} mm")

# Reconstituer les grilles 2D originales pour l'interpolation si elles ont été stockées en 1D
# Trouvez les limites de la ROI à partir des données chargées
x_roi_min_loaded = np.min(X_interp_flat)
x_roi_max_loaded = np.max(X_interp_flat)
y_roi_min_loaded = np.min(Y_interp_flat)
y_roi_max_loaded = np.max(Y_interp_flat)

# Créez des axes réguliers à partir des min/max et h_interp
x_vals_unique = np.unique(X_interp_flat)
y_vals_unique = np.unique(Y_interp_flat)

# Assurez-vous que les grilles sont bien triées pour RegularGridInterpolator
x_vals_unique.sort()
y_vals_unique.sort()

# Reconstruire les tableaux 2D Ex_interp et Ey_interp
# Nous devons d'abord créer une grille complète sur la ROI
X_full_grid, Y_full_grid = np.meshgrid(x_vals_unique, y_vals_unique)
Ex_full_grid = np.full(X_full_grid.shape, np.nan)
Ey_full_grid = np.full(Y_full_grid.shape, np.nan)

# Remplir les valeurs existantes sur la grille complète
# Créer un dictionnaire pour un accès rapide aux indices
point_to_idx = {}
for i in range(len(x_vals_unique)):
    for j in range(len(y_vals_unique)):
        point_to_idx[(round(x_vals_unique[i], 5), round(y_vals_unique[j], 5))] = (j, i) # (row, col)

for i in range(len(X_interp_flat)):
    x_coord = round(X_interp_flat[i], 5)
    y_coord = round(Y_interp_flat[i], 5)
    if (x_coord, y_coord) in point_to_idx:
        row, col = point_to_idx[(x_coord, y_coord)]
        Ex_full_grid[row, col] = Ex_interp_flat[i]
        Ey_full_grid[row, col] = Ey_interp_flat[i]

# Pour l'interpolation du champ E(x,y)
# Les valeurs NaN peuvent poser problème, nous allons les remplacer par 0 pour l'interpolateur
# car un champ de 0 est une approximation raisonnable en dehors de la zone calculée.
Ex_full_grid = np.nan_to_num(Ex_full_grid, nan=0.0)
Ey_full_grid = np.nan_to_num(Ey_full_grid, nan=0.0)

# Créer les fonctions d'interpolation pour Ex et Ey
interp_Ex = RegularGridInterpolator((y_vals_unique, x_vals_unique), Ex_full_grid, method='linear', bounds_error=False, fill_value=0.0)
interp_Ey = RegularGridInterpolator((y_vals_unique, x_vals_unique), Ey_full_grid, method='linear', bounds_error=False, fill_value=0.0)


# --- 2. Paramètres de la simulation de l'électron ---
print("\n=== Paramètres de la simulation de l'électron ===")

# Position initiale de l'électron (sur la bande centrale)
# Trouver la position x du centre de la bande centrale
if nb_bandes % 2 != 0: # Si le nombre de bandes est impair, il y a une bande centrale
    center_band_index = nb_bandes // 2
    x_start_electron = (-total_width_bottom / 2) + center_band_index * (band_width + gap) + band_width / 2
else: # Si le nombre de bandes est pair, prendre le centre du gap central
    x_start_electron = (-total_width_bottom / 2) + (nb_bandes / 2) * (band_width + gap) - gap / 2

y_start_electron = -plan_gap / 2 # Sur le plan inférieur
initial_position = np.array([x_start_electron, y_start_electron]) * 1e-3 # Convertir en mètres
initial_velocity = np.array([0.0, 0.0]) # Sans vitesse initiale

print(f"Position de départ de l'électron (x, y) : ({initial_position[0]*1e3:.3f} mm, {initial_position[1]*1e3:.3f} mm)")

# Pas de temps et durée de simulation
dt = input_default("Pas de temps de simulation (secondes, ex: 1e-12)", 1e-12)
simulation_duration = input_default("Durée totale de la simulation (secondes, ex: 1e-9)", 1e-9)
num_steps = int(simulation_duration / dt)

# --- 3. Simulation du mouvement de l'électron ---
print(f"\nDébut de la simulation pour {num_steps} pas de temps...")

# Liste pour stocker la trajectoire
trajectory_data = [] # [time, x_pos, y_pos, vx, vy]

current_position = initial_position.copy()
current_velocity = initial_velocity.copy()
current_time = 0.0

# Ajout de la première entrée
trajectory_data.append([current_time, current_position[0], current_position[1],
                         current_velocity[0], current_velocity[1]])

# Boucle de simulation
for i in range(num_steps):
    # Obtenir le champ électrique à la position actuelle (convertir en mm pour l'interpolateur)
    # Assurez-vous de rester dans les limites définies par l'interpolateur
    pos_mm_x = current_position[0] * 1e3
    pos_mm_y = current_position[1] * 1e3

    # Gérer les cas où l'électron sort des limites de la ROI ou du cercle
    if not (x_roi_min_loaded <= pos_mm_x <= x_roi_max_loaded and
            y_roi_min_loaded <= pos_mm_y <= y_roi_max_loaded):
        # Si l'électron quitte la ROI, ou le cercle, on arrête la simulation
        # et on assume un champ nul ou une sortie du domaine d'intérêt
        print(f"Électron sorti de la ROI à t={current_time:.2e}s, position=({pos_mm_x:.3f}mm, {pos_mm_y:.3f}mm). Arrêt de la simulation.")
        break

    E_x = interp_Ex([[pos_mm_y, pos_mm_x]])[0] # Note: RegularGridInterpolator attend (y, x)
    E_y = interp_Ey([[pos_mm_y, pos_mm_x]])[0]

    # Force électrique F = qE
    force_x = CHARGE_ELECTRON * E_x * 1e3 # E est en V/mm, besoin de le convertir en V/m
    force_y = CHARGE_ELECTRON * E_y * 1e3 # V/mm * 1000 mm/m = V/m

    # Accélération a = F/m
    acceleration_x = force_x / MASSE_ELECTRON
    acceleration_y = force_y / MASSE_ELECTRON

    # Nouvelle vitesse (v = v0 + at)
    current_velocity[0] += acceleration_x * dt
    current_velocity[1] += acceleration_y * dt

    # Nouvelle position (x = x0 + vt)
    current_position[0] += current_velocity[0] * dt
    current_position[1] += current_velocity[1] * dt

    current_time += dt

    # Stocker la position et la vitesse (conversion en mm pour la position)
    trajectory_data.append([current_time, current_position[0], current_position[1],
                             current_velocity[0], current_velocity[1]])

    if (i + 1) % (num_steps // 10 if num_steps >= 10 else 1) == 0 or i == num_steps - 1:
        print(f"  Progression : {((i+1)/num_steps*100):.1f}%")

print(f"Simulation terminée en {len(trajectory_data)-1} pas de temps.")

# Convertir la liste en tableau NumPy pour faciliter l'analyse
trajectory_data = np.array(trajectory_data)

# --- 4. Sauvegarde des données de trajectoire ---
np.savez("electron_trajectory.npz",
         time=trajectory_data[:, 0],
         x_pos_m=trajectory_data[:, 1],
         y_pos_m=trajectory_data[:, 2],
         vx_m_s=trajectory_data[:, 3],
         vy_m_s=trajectory_data[:, 4],
         dt=dt,
         initial_x_mm=initial_position[0]*1e3,
         initial_y_mm=initial_position[1]*1e3
)
print("\nDonnées de trajectoire de l'électron sauvegardées dans 'electron_trajectory.npz'")

# --- 5. Affichage de la trajectoire ---
print("\n=== Affichage de la trajectoire de l'électron ===")

fig, ax = plt.subplots(figsize=(10, 8))

# Define display_padding for the plotting limits
display_padding = h_interp * 2 # A small visual padding based on the interpolation discretization

# Afficher la magnitude du champ électrique en arrière-plan (interpolation pour l'affichage)
# Reconstruire Ex_full_grid et Ey_full_grid pour le plotting
Ex_plot = interp_Ex(np.vstack([Y_full_grid.ravel(), X_full_grid.ravel()]).T).reshape(Y_full_grid.shape)
Ey_plot = interp_Ey(np.vstack([Y_full_grid.ravel(), X_full_grid.ravel()]).T).reshape(Y_full_grid.shape)
E_magnitude_plot = np.sqrt(Ex_plot**2 + Ey_plot**2)

# Masquer les valeurs hors ROI pour l'affichage
mask_roi_plot = (X_full_grid >= x_roi_min_loaded) & (X_full_grid <= x_roi_max_loaded) & \
                  (Y_full_grid >= y_roi_min_loaded) & (Y_full_grid <= y_roi_max_loaded)
E_magnitude_plot[~mask_roi_plot] = np.nan
Ex_plot[~mask_roi_plot] = np.nan
Ey_plot[~mask_roi_plot] = np.nan


# Afficher le champ en contourf
c = ax.contourf(X_full_grid, Y_full_grid, E_magnitude_plot, levels=100, cmap='viridis')
plt.colorbar(c, ax=ax, label='|E| (V/mm)')

# Afficher la trajectoire de l'électron (conversion en mm pour l'affichage)
ax.plot(trajectory_data[:, 1] * 1e3, trajectory_data[:, 2] * 1e3, 'r-', linewidth=2, label='Trajectoire de l\'électron')
ax.plot(initial_position[0] * 1e3, initial_position[1] * 1e3, 'ro', markersize=8, label='Départ')
ax.plot(trajectory_data[-1, 1] * 1e3, trajectory_data[-1, 2] * 1e3, 'rx', markersize=8, label='Arrivée')

# Afficher les lignes des plans
ax.axhline(plan_gap / 2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, label='Plan supérieur (100V)')
ax.axhline(-plan_gap / 2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, label='Plans inférieurs (0V)')

# Ajouter des flèches de champ électrique pour le contexte
ax.streamplot(X_full_grid, Y_full_grid, Ex_plot, Ey_plot, color='k', density=1.5, linewidth=0.8, arrowsize=1.5)


ax.set_title("Trajectoire de l'électron dans le champ électrique")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()

# Définir les limites d'affichage pour la ROI (avec un petit padding visuel)
ax.set_xlim(x_roi_min_loaded - display_padding, x_roi_max_loaded + display_padding)
ax.set_ylim(y_roi_min_loaded - display_padding, y_roi_max_loaded + display_padding)


plt.tight_layout()
plt.show()