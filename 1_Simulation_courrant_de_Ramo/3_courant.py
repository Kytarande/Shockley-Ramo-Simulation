import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# --- Constantes physiques ---
CHARGE_ELECTRON = -1.602e-19  # Charge de l'électron en Coulombs (C)

print("=== Calcul et Affichage des Courants et Charges de Ramos ===")

# --- 1. Charger toutes les données nécessaires ---
try:
    # Charger les données du champ électrique (pour les grilles et paramètres)
    field_data = np.load("champ_electrique_donnees_ROI.npz", allow_pickle=True)

    # Charger les données de la trajectoire de l'électron
    trajectory_data = np.load("electron_trajectory.npz")

except FileNotFoundError:
    print("Erreur : Un ou plusieurs fichiers de données sont introuvables.")
    print("Assurez-vous d'avoir d'abord exécuté le script de simulation du champ électrique et le script de simulation de la trajectoire de l'électron.")
    exit()

print("\nDonnées de champ électrique et de trajectoire chargées avec succès.")

# Récupérer les informations de la grille spatiale et de la ROI
X_interp_flat = field_data['X_interp']
Y_interp_flat = field_data['Y_interp']
h_interp = field_data['h_interp']
nb_bandes = field_data['nb_bandes']
band_width = field_data['band_width']
gap = field_data['gap']
total_width_bottom = field_data['total_width_bottom']
plan_gap = field_data['plan_gap']

# Récupérer les données de la trajectoire de l'électron
times = trajectory_data['time']
x_pos_m = trajectory_data['x_pos_m']
y_pos_m = trajectory_data['y_pos_m']
vx_m_s = trajectory_data['vx_m_s']
vy_m_s = trajectory_data['vy_m_s']
dt_simulation = trajectory_data['dt'] # Récupérer le dt de la simulation de trajectoire

# --- 2. Préparer les interpolateurs pour les champs de Ramos ---

# Reconstituer les axes uniques pour RegularGridInterpolator
x_vals_unique = np.unique(X_interp_flat)
y_vals_unique = np.unique(Y_interp_flat)
x_vals_unique.sort()
y_vals_unique.sort()

# Dictionnaire pour stocker les interpolateurs de Ramos pour chaque électrode
ramos_interpolators = {}

# Récupérer les données des champs de Ramos
ramos_field_data = field_data['ramos_data'].item() # .item() pour convertir le tableau de dtype=object en dict

# Fonction utilitaire pour créer un interpolateur pour Ex et Ey
def create_ramos_interpolators(data_dict, x_unique, y_unique):
    # Reconstruire les grilles 2D Ex_ramos et Ey_ramos à partir des données 1D filtrées
    X_full_grid, Y_full_grid = np.meshgrid(x_unique, y_unique)
    Ex_ramos_full_grid = np.full(X_full_grid.shape, np.nan)
    Ey_ramos_full_grid = np.full(Y_full_grid.shape, np.nan)

    # Créer un dictionnaire pour un accès rapide aux indices (row, col)
    point_to_idx = {}
    for i in range(len(x_unique)):
        for j in range(len(y_unique)):
            point_to_idx[(round(x_unique[i], 5), round(y_unique[j], 5))] = (j, i) # (row, col)

    for i in range(len(data_dict['X'])):
        x_coord = round(data_dict['X'][i], 5)
        y_coord = round(data_dict['Y'][i], 5)
        if (x_coord, y_coord) in point_to_idx:
            row, col = point_to_idx[(x_coord, y_coord)]
            Ex_ramos_full_grid[row, col] = data_dict['Ex'][i]
            Ey_ramos_full_grid[row, col] = data_dict['Ey'][i]

    # Remplacer les NaN par 0 pour l'interpolateur
    Ex_ramos_full_grid = np.nan_to_num(Ex_ramos_full_grid, nan=0.0)
    Ey_ramos_full_grid = np.nan_to_num(Ey_ramos_full_grid, nan=0.0)

    interp_Ex = RegularGridInterpolator((y_unique, x_unique), Ex_ramos_full_grid, method='linear', bounds_error=False, fill_value=0.0)
    interp_Ey = RegularGridInterpolator((y_unique, x_unique), Ey_ramos_full_grid, method='linear', bounds_error=False, fill_value=0.0)
    return interp_Ex, interp_Ey

# Créer les interpolateurs pour le plan supérieur
ramos_interpolators['plane_100'] = create_ramos_interpolators(ramos_field_data['plane_100'], x_vals_unique, y_vals_unique)
print("Interpolateurs pour le champ de Ramos (plan supérieur) créés.")

# Créer les interpolateurs pour chaque bande inférieure
for j in range(nb_bandes):
    key = f'band_{j+1}'
    ramos_interpolators[key] = create_ramos_interpolators(ramos_field_data[key], x_vals_unique, y_vals_unique)
    print(f"Interpolateurs pour le champ de Ramos (bande {j+1}) créés.")

# Déterminer les clés des électrodes pour l'affichage : plan, bande centrale, bande gauche, bande droite
electrode_keys_to_display = ['plane_100'] # Toujours inclure le plan supérieur

# Trouver la bande centrale (la plus proche de x=0) ou gérer les cas pairs
if nb_bandes % 2 != 0: # Impair: il y a une vraie bande centrale
    central_band_idx = nb_bandes // 2
    central_band_key = f'band_{central_band_idx + 1}'
else: # Pair: Pas de bande centrale unique, on prend la première des deux du milieu
    central_band_idx = nb_bandes // 2 - 1 # Index de la bande la plus à gauche des deux du milieu
    central_band_key = f'band_{central_band_idx + 1}'

electrode_keys_to_display.append(central_band_key)


# Bande gauche : si nb_bandes > 1 et la bande centrale n'est pas la première
if nb_bandes > 1 and central_band_idx > 0:
    left_band_key = f'band_{central_band_idx}' # Bande à gauche de la centrale
    electrode_keys_to_display.append(left_band_key)

# Bande droite : si nb_bandes > 1 and la bande centrale n'est pas la dernière
if nb_bandes > 1 and central_band_idx < nb_bandes - 1:
    right_band_key = f'band_{central_band_idx + 2}' # Bande à droite de la centrale
    electrode_keys_to_display.append(right_band_key)

# Éviter les doublons si 'gauche' ou 'droite' pointe vers la 'centrale' en cas de nb_bandes faibles
electrode_keys_to_display = list(dict.fromkeys(electrode_keys_to_display)) # Supprime les doublons en gardant l'ordre

# --- 3. Calculer les courants induits pour chaque électrode ---
print("\nCalcul des courants induits...")

currents_data = {} # Stockera les listes de courants pour chaque électrode

for key in electrode_keys_to_display:
    currents_data[key] = []

for i in range(len(times)):
    pos_mm_x = x_pos_m[i] * 1e3 # Convertir la position en mm
    pos_mm_y = y_pos_m[i] * 1e3

    velocity_vec = np.array([vx_m_s[i], vy_m_s[i]])

    for electrode_name in electrode_keys_to_display:
        interp_Ex_ramos, interp_Ey_ramos = ramos_interpolators[electrode_name]

        # Obtenir le champ de Ramos à la position de l'électron
        # RegularGridInterpolator attend (y, x) pour l'évaluation
        E_ramos_x = interp_Ex_ramos([[pos_mm_y, pos_mm_x]])[0]
        E_ramos_y = interp_Ey_ramos([[pos_mm_y, pos_mm_x]])[0]

        E_ramos_vec = np.array([E_ramos_x, E_ramos_y]) * 1e3 # Convertir E_ramos de V/mm en V/m

        # Calcul du produit scalaire : v . E_ramos
        dot_product = np.dot(velocity_vec, E_ramos_vec)

        # Calcul du courant I(t) = q * (v . E_ramos)
        current = CHARGE_ELECTRON * dot_product
        currents_data[electrode_name].append(current)

print("Calcul des courants terminé.")

# --- 4. Affichage des courants et charges en fonction du temps ---
print("\n=== Affichage des Courants et Charges de Ramos ===")

# Dictionnaire pour les titres lisibles des électrodes
display_titles = {
    'plane_100': 'Plan Supérieur',
}
if 'band_1' in ramos_interpolators: # Si au moins une bande existe
    # On assigne les titres basés sur les clés réelles déterminées pour l'affichage
    if nb_bandes % 2 != 0:
        display_titles[central_band_key] = f'Bande Centrale (bande {central_band_idx + 1})'
    else:
        # Pour les nombres pairs de bandes, on indique que c'est la bande "la plus" centrale
        display_titles[central_band_key] = f'Bande (la plus) Centrale (bande {central_band_idx + 1})'

    # Gestion des cas où 'left_band_key' ou 'right_band_key' est défini
    if 'left_band_key' in locals() and left_band_key != central_band_key:
        display_titles[left_band_key] = f'Bande Gauche (bande {central_band_idx})'
    if 'right_band_key' in locals() and right_band_key != central_band_key and ('left_band_key' not in locals() or right_band_key != left_band_key):
        display_titles[right_band_key] = f'Bande Droite (bande {central_band_idx + 2})'
    elif nb_bandes == 2 and central_band_key == 'band_1': # Cas spécifique pour 2 bandes
        display_titles['band_1'] = 'Bande Gauche (bande 1)'
        display_titles['band_2'] = 'Bande Droite (bande 2)'
    elif nb_bandes == 1:
        display_titles['band_1'] = 'Bande Unique'

for electrode_name in electrode_keys_to_display:
    currents_array = np.array(currents_data[electrode_name])

    # Calcul de l'intégrale du courant pour obtenir la charge
    integrated_charge = np.cumsum(currents_array * dt_simulation)

    # Normalisation de la charge intégrée par la charge d'un électron
    normalized_integrated_charge = integrated_charge / CHARGE_ELECTRON # Corrected normalization

    # --- Diagnostic : Afficher les plages de valeurs ---
    print(f"\n--- Plages de valeurs pour '{display_titles.get(electrode_name, electrode_name)}' ---")
    print(f"Courant (A): Min={np.min(currents_array):.2e}, Max={np.max(currents_array):.2e}, Avg={np.mean(currents_array):.2e}")
    print(f"Charge Normalisée: Min={np.min(normalized_integrated_charge):.2e}, Max={np.max(normalized_integrated_charge):.2e}")


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'Courant et Charge induits sur {display_titles.get(electrode_name, electrode_name)}', fontsize=16)

    # Plot du Courant
    ax1.plot(times * 1e9, currents_array * 1e9, label='Courant induit', color='blue') # nA
    ax1.set_ylabel("Courant (nA)")
    ax1.set_title("Courant en fonction du temps")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.7)


    # Plot de la Charge Intégrée Normalisée
    ax2.plot(times * 1e9, normalized_integrated_charge, label='Charge intégrée normalisée', color='red')
    ax2.set_xlabel("Temps (ns)")
    ax2.set_ylabel("Charge Normalisée (unités de charge d'électron)")
    ax2.set_title("Charge intégrée normalisée en fonction du temps")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    # La ligne de référence à 1 ou -1 dépend de la charge de l'électron
    ax2.axhline(1 if CHARGE_ELECTRON > 0 else -1, color='green', linestyle=':', linewidth=0.7, label='$\pm 1$ charge d\'électron')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuste le layout pour le titre global
    plt.legend()
    plt.show()