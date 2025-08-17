import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.interpolate import griddata

# --- Fonction d'entrée sécurisée avec valeur par défaut ---
def input_default(prompt, default, type_=float):
    val = input(f"{prompt} [{default}] : ").strip()
    return type_(val) if val else default

# --- Paramètres entrés par l'utilisateur ---
print("=== Configuration du système ===")

# Bandes inférieures
nb_bandes = int(input_default("Nombre de bandes à 0V", 3, int))
band_width = input_default("Largeur d'une bande (mm)", 1.84)
gap = input_default("Gap entre bandes (mm)", 0.35)

# Largeur totale des gap et bandes
total_width_bottom = round(nb_bandes * band_width +
                            (nb_bandes - 1) * gap, 2)

# Plan supérieur
width_100 = input_default("Largeur du plan supérieur (mm)", total_width_bottom)

# Écart entre les deux plans (haut/bas)
plan_gap = input_default("Distance entre les plans (mm)", 1.0)

# Discrétisation de la simulation globale
h = input_default("Discrétisation spatiale pour la simulation globale (mm)", 0.1)

# Discrétisation pour l'interpolation et l'affichage (ROI)
h_interp_default = h / 2
h_interp = input_default(f"Discrétisation spatiale pour l'interpolation (mm) (par défaut: {h_interp_default:.2f})", h_interp_default)


# Taille minimale du rayon
rayon_min = round( np.sqrt( (max(width_100,total_width_bottom)/2)**2 + (plan_gap/2)**2 ), 2)

# Rayon du cercle
R = input_default(f"Rayon du cercle (mm) (>= {rayon_min:.2f})", max(20.0, rayon_min))
while R <= rayon_min:
    print(f"Le rayon est trop petit, il doit être au minimum de {rayon_min:.2f} mm.")
    R = input_default(f"Rayon du cercle (mm) (>= {rayon_min:.2f})", rayon_min)

# Calcul de la grille globale
L = 2 * R
res = int(L / h) + 1
print("\n=== Récapitulatif de la configuration ===")
print(f"Taille de la grille globale : {res} x {res}")
print(f"Largeur totale des bandes inférieures : {total_width_bottom:.2f} mm")
print(f"Largeur du plan supérieur : {width_100:.2f} mm")
print(f"Discrétisation globale : {h:.2f} mm")
print(f"Discrétisation interpolée (ROI) : {h_interp:.2f} mm")
print(f"Rayon du cercle : {R:.2f} mm")
print(f"Écart entre les plans : {plan_gap:.2f} mm")

# Taux de convergence
tolerance = input_default("\nTolérance de convergence (ex: 1e-4)", 1e-4)

print("\nLancement du calcul...\n")

# --- Grille d'espace globale ---
x = np.linspace(-R, R, res)
y = np.linspace(-R, R, res)
X, Y = np.meshgrid(x, y)

# --- Masque du cercle (pour la simulation globale) ---
mask_circle = X**2 + Y**2 <= R**2

# --- Initialisation du potentiel global ---
V = np.zeros_like(X)
V[~mask_circle] = np.nan  # Hors du cercle = non calculé

# --- Électrode à 100 V (plan supérieur) ---
x_min_100 = -width_100 / 2
x_max_100 = width_100 / 2
y_100 = plan_gap / 2 # Centre du plan supérieur

electrode_100 = (
    (np.abs(Y - y_100) < h / 2) & # Identifie les points sur le plan supérieur
    (X >= x_min_100) & (X <= x_max_100)
)

# --- Bandes à 0 V (plans inférieurs) ---
x_left = -total_width_bottom / 2
y_0 = -plan_gap / 2 # Centre des plans inférieurs
electrode_0_list = [] # Stocke les masques de chaque bande individuelle

for i in range(nb_bandes):
    xi_min = x_left + i * (band_width + gap)
    xi_max = xi_min + band_width
    mask = (
        (np.abs(Y - y_0) < h / 2) & # Identifie les points sur la bande
        (X >= xi_min) & (X <= xi_max)
    )
    electrode_0_list.append(mask)

# Combinaison de toutes les électrodes à 0V pour la simulation principale
electrode_0_combined = np.zeros_like(X, dtype=bool)
for mask in electrode_0_list:
    electrode_0_combined |= mask

# --- Conditions initiales pour la simulation principale ---
V[electrode_100] = 100
V[electrode_0_combined] = 0
V[~mask_circle] = 0  # Bord du cercle à 0 V

# --- Masque des points figés pour la simulation principale ---
fixed_mask = electrode_100 | electrode_0_combined | ~mask_circle

# --- Relaxation pondérée (méthode des 9 points) pour la simulation principale ---
iteration = 0
print("\nDébut de la relaxation pour le champ principal...")
while True:
    V_old = V.copy()

    # Moyenne orthogonale
    Vpp = (
        np.roll(V, 1, axis=0) + np.roll(V, -1, axis=0) +
        np.roll(V, 1, axis=1) + np.roll(V, -1, axis=1)
    ) / 4

    # Moyenne diagonale
    Vs = (
        np.roll(np.roll(V, 1, axis=0), 1, axis=1) +
        np.roll(np.roll(V, 1, axis=0), -1, axis=1) +
        np.roll(np.roll(V, -1, axis=0), 1, axis=1) +
        np.roll(np.roll(V, -1, axis=0), -1, axis=1)
    ) / 4

    V_new = 0.8 * Vpp + 0.2 * Vs
    V[~fixed_mask] = V_new[~fixed_mask]

    sigma = np.nanmax(np.abs(V[~fixed_mask] - V_old[~fixed_mask])) # Vérifie uniquement les points non figés

    iteration += 1
    if iteration % 100 == 0:
        convergence_pourcentage = (tolerance / sigma) * 100 if sigma != 0 else 100
        print(f"  Iteration {iteration}, delta = {sigma:.2e}, pourcentage = {convergence_pourcentage:.4f} %")

    if sigma < tolerance:
        print(f"  Iteration {iteration}, delta = {sigma:.2e}, pourcentage = 100 %")
        print(f"\nConvergence atteinte en {iteration} itérations pour le champ principal.")
        break

# --- Calcul et affichage pour la région entre les plans ---
print("\n=== Analyse et affichage de la région entre les plans ===")

# Définition de la Région d'Intérêt (ROI) pour l'affichage et la sauvegarde
# STRICTEMENT entre les plans, sans aucun padding supplémentaire
x_roi_min = min(-total_width_bottom / 2, -width_100 / 2)
x_roi_max = max(total_width_bottom / 2, width_100 / 2)
y_roi_min = -plan_gap / 2
y_roi_max = plan_gap / 2

# Créer une grille pour l'interpolation dans la ROI
x_interp_grid = np.arange(x_roi_min, x_roi_max + h_interp, h_interp)
y_interp_grid = np.arange(y_roi_min, y_roi_max + h_interp, h_interp)
X_interp, Y_interp = np.meshgrid(x_interp_grid, y_interp_grid)

# Préparation des points de la grille originale pour l'interpolation
points_global_grid = np.array([X.flatten(), Y.flatten()]).T
values_global_grid = V.flatten() # Potentiel de la simulation principale

# Interpolation du potentiel sur la grille fine de la ROI
V_interp = griddata(points_global_grid, values_global_grid, (X_interp, Y_interp), method='cubic')

# Calcul du champ électrique interpolé dans la ROI
Ey_interp, Ex_interp = np.gradient(-V_interp, h_interp, h_interp)

# --- Masquage des points hors ROI après interpolation ---
# Cette étape est cruciale pour garantir que seuls les points dans la ROI sont affichés/enregistrés
mask_roi_interp = (X_interp >= x_roi_min) & (X_interp <= x_roi_max) & \
                  (Y_interp >= y_roi_min) & (Y_interp <= y_roi_max)

V_interp[~mask_roi_interp] = np.nan
Ex_interp[~mask_roi_interp] = np.nan
Ey_interp[~mask_roi_interp] = np.nan

# --- Affichage du potentiel et du champ électrique dans la ROI ---
fig_roi, axes_roi = plt.subplots(1, 2, figsize=(14, 6))
fig_roi.suptitle("Potentiel et Champ Électrique (Région entre les plans)")

# Padding pour les limites d'affichage (pour éviter le rognage des contours à la limite exacte)
display_padding = h_interp * 2 # Un petit padding basé sur la discrétisation

# Potentiel
c_roi = axes_roi[0].contourf(X_interp, Y_interp, V_interp, levels=100, cmap='plasma')
axes_roi[0].set_title("Potentiel (V)")
axes_roi[0].set_aspect('equal')
plt.colorbar(c_roi, ax=axes_roi[0], label='Potentiel (V)')
axes_roi[0].set_xlabel("x (mm)")
axes_roi[0].set_ylabel("y (mm)")
axes_roi[0].set_xlim(x_roi_min - display_padding, x_roi_max + display_padding)
axes_roi[0].set_ylim(y_roi_min - display_padding, y_roi_max + display_padding)
# Ajout des lignes des plans pour la clarté
axes_roi[0].axhline(y_100, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
axes_roi[0].axhline(y_0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)


# Champ Électrique
strm_roi = axes_roi[1].streamplot(X_interp, Y_interp, Ex_interp, Ey_interp,
                                  color=np.sqrt(Ex_interp**2 + Ey_interp**2), cmap='viridis', density=1.5)
axes_roi[1].set_title("Champ Électrique |E| (V/mm)")
axes_roi[1].set_aspect('equal')
plt.colorbar(strm_roi.lines, ax=axes_roi[1], label='|E| (V/mm)')
axes_roi[1].set_xlabel("x (mm)")
axes_roi[1].set_ylabel("y (mm)")
axes_roi[1].set_xlim(x_roi_min - display_padding, x_roi_max + display_padding)
axes_roi[1].set_ylim(y_roi_min - display_padding, y_roi_max + display_padding)
# Ajout des lignes des plans pour la clarté
axes_roi[1].axhline(y_100, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
axes_roi[1].axhline(y_0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Calcul et affichage des champs de Ramos ---
print("\n=== Calcul des champs de Ramos ===")

ramos_data = {}

# --- Ramos pour le plan supérieur (Electrode à 1V, autres à 0V) ---
print("\nDébut de la relaxation pour le champ de Ramos (plan supérieur)...")
V_ramos_100 = np.zeros_like(X)
V_ramos_100[~mask_circle] = np.nan # Hors du cercle = non calculé

# Définition des conditions aux limites pour Ramos: plan sup à 1V, autres à 0V
V_ramos_100[electrode_100] = 1 # Le plan supérieur à 1V
V_ramos_100[electrode_0_combined] = 0 # Toutes les bandes inférieures à 0V
V_ramos_100[~mask_circle] = 0 # Le bord du cercle à 0V

# Masque des points figés pour ce calcul de Ramos
fixed_mask_ramos_100 = electrode_100 | electrode_0_combined | ~mask_circle

# Relaxation pour Ramos (plan supérieur)
iteration_ramos = 0
while True:
    V_old_ramos = V_ramos_100.copy()

    Vpp_ramos = (
        np.roll(V_ramos_100, 1, axis=0) + np.roll(V_ramos_100, -1, axis=0) +
        np.roll(V_ramos_100, 1, axis=1) + np.roll(V_ramos_100, -1, axis=1)
    ) / 4

    Vs_ramos = (
        np.roll(np.roll(V_ramos_100, 1, axis=0), 1, axis=1) +
        np.roll(np.roll(V_ramos_100, 1, axis=0), -1, axis=1) +
        np.roll(np.roll(V_ramos_100, -1, axis=0), 1, axis=1) +
        np.roll(np.roll(V_ramos_100, -1, axis=0), -1, axis=1)
    ) / 4

    V_new_ramos = 0.8 * Vpp_ramos + 0.2 * Vs_ramos
    V_ramos_100[~fixed_mask_ramos_100] = V_new_ramos[~fixed_mask_ramos_100]

    sigma_ramos = np.nanmax(np.abs(V_ramos_100[~fixed_mask_ramos_100] - V_old_ramos[~fixed_mask_ramos_100]))
    iteration_ramos += 1
    if iteration_ramos % 100 == 0:
        convergence_pourcentage_ramos = (tolerance / sigma_ramos) * 100 if sigma_ramos != 0 else 100
        print(f"  Iteration {iteration_ramos}, delta = {sigma_ramos:.2e}, pourcentage = {convergence_pourcentage_ramos:.4f} %")

    if sigma_ramos < tolerance:
        print(f"  Iteration {iteration_ramos}, delta = {sigma_ramos:.2e}, pourcentage = 100 %")
        print(f"Convergence atteinte pour le plan supérieur en {iteration_ramos} itérations.")
        break

# Calcul du champ électrique Ramos (plan supérieur)
Ey_ramos_100, Ex_ramos_100 = np.gradient(-V_ramos_100, h, h)

# Interpolation des données Ramos (plan supérieur) sur la ROI
V_ramos_100_interp = griddata(points_global_grid, V_ramos_100.flatten(), (X_interp, Y_interp), method='cubic')
Ey_ramos_100_interp, Ex_ramos_100_interp = np.gradient(-V_ramos_100_interp, h_interp, h_interp)

# --- Masquage des points hors ROI après interpolation pour Ramos ---
V_ramos_100_interp[~mask_roi_interp] = np.nan
Ex_ramos_100_interp[~mask_roi_interp] = np.nan
Ey_ramos_100_interp[~mask_roi_interp] = np.nan


# Filtrage des données 1D pour l'enregistrement
valid_mask_1d_ramos_100 = ~np.isnan(V_ramos_100_interp.flatten())
ramos_data['plane_100'] = {
    'V': V_ramos_100_interp.flatten()[valid_mask_1d_ramos_100],
    'Ex': Ex_ramos_100_interp.flatten()[valid_mask_1d_ramos_100],
    'Ey': Ey_ramos_100_interp.flatten()[valid_mask_1d_ramos_100],
    'X': X_interp.flatten()[valid_mask_1d_ramos_100],
    'Y': Y_interp.flatten()[valid_mask_1d_ramos_100]
}

# --- Affichage du potentiel et du champ électrique Ramos (plan supérieur) ---
fig_ramos_100, axes_ramos_100 = plt.subplots(1, 2, figsize=(14, 6))
fig_ramos_100.suptitle("Champ de Ramos : Plan supérieur (1V) (Région entre les plans)")

c_ramos_100 = axes_ramos_100[0].contourf(X_interp, Y_interp, V_ramos_100_interp, levels=100, cmap='plasma')
axes_ramos_100[0].set_title("Potentiel de Ramos (V)")
axes_ramos_100[0].set_aspect('equal')
plt.colorbar(c_ramos_100, ax=axes_ramos_100[0], label='Potentiel (V)')
axes_ramos_100[0].set_xlabel("x (mm)")
axes_ramos_100[0].set_ylabel("y (mm)")
axes_ramos_100[0].set_xlim(x_roi_min - display_padding, x_roi_max + display_padding)
axes_ramos_100[0].set_ylim(y_roi_min - display_padding, y_roi_max + display_padding)
axes_ramos_100[0].axhline(y_100, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
axes_ramos_100[0].axhline(y_0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)


strm_ramos_100 = axes_ramos_100[1].streamplot(X_interp, Y_interp, Ex_ramos_100_interp, Ey_ramos_100_interp,
                                             color=np.sqrt(Ex_ramos_100_interp**2 + Ey_ramos_100_interp**2), cmap='viridis', density=1.5)
axes_ramos_100[1].set_title("Champ Électrique de Ramos |E| (V/mm)")
axes_ramos_100[1].set_aspect('equal')
plt.colorbar(strm_ramos_100.lines, ax=axes_ramos_100[1], label='|E| (V/mm)')
axes_ramos_100[1].set_xlabel("x (mm)")
axes_ramos_100[1].set_ylabel("y (mm)")
axes_ramos_100[1].set_xlim(x_roi_min - display_padding, x_roi_max + display_padding)
axes_ramos_100[1].set_ylim(y_roi_min - display_padding, y_roi_max + display_padding)
axes_ramos_100[1].axhline(y_100, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
axes_ramos_100[1].axhline(y_0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# --- Ramos pour les bandes inférieures (chaque bande à 1V, autres à 0V) ---
for j, electrode_0_mask in enumerate(electrode_0_list):
    print(f"\nDébut de la relaxation pour le champ de Ramos (bande inférieure {j+1})...")
    V_ramos_band = np.zeros_like(X)
    V_ramos_band[~mask_circle] = np.nan # Hors du cercle = non calculé

    # Construction du masque des points figés pour ce calcul de Ramos
    # Le plan supérieur est à 0V, le cercle à 0V, les autres bandes à 0V
    current_fixed_mask_ramos_band = electrode_100.copy() # Plan supérieur à 0V
    current_fixed_mask_ramos_band |= ~mask_circle # Bord du cercle à 0V
    current_fixed_mask_ramos_band |= electrode_0_mask # La bande actuelle à 1V

    # Toutes les autres bandes à 0V sont également figées à 0V
    for k, other_band_mask in enumerate(electrode_0_list):
        if k != j:
            current_fixed_mask_ramos_band |= other_band_mask

    # Définition des potentiels pour ce calcul de Ramos
    V_ramos_band[electrode_100] = 0 # Plan supérieur à 0V
    for k, other_band_mask in enumerate(electrode_0_list):
        if k == j:
            V_ramos_band[other_band_mask] = 1 # Bande actuelle à 1V
        else:
            V_ramos_band[other_band_mask] = 0 # Autres bandes à 0V
    V_ramos_band[~mask_circle] = 0 # Bord du cercle à 0V

    # Relaxation pour Ramos (bande individuelle)
    iteration_ramos_band = 0
    while True:
        V_old_ramos_band = V_ramos_band.copy()

        Vpp_ramos_band = (
            np.roll(V_ramos_band, 1, axis=0) + np.roll(V_ramos_band, -1, axis=0) +
            np.roll(V_ramos_band, 1, axis=1) + np.roll(V_ramos_band, -1, axis=1)
        ) / 4

        Vs_ramos_band = (
            np.roll(np.roll(V_ramos_band, 1, axis=0), 1, axis=1) +
            np.roll(np.roll(V_ramos_band, 1, axis=0), -1, axis=1) +
            np.roll(np.roll(V_ramos_band, -1, axis=0), 1, axis=1) +
            np.roll(np.roll(V_ramos_band, -1, axis=0), -1, axis=1)
        ) / 4

        V_new_ramos_band = 0.8 * Vpp_ramos_band + 0.2 * Vs_ramos_band
        V_ramos_band[~current_fixed_mask_ramos_band] = V_new_ramos_band[~current_fixed_mask_ramos_band]

        sigma_ramos_band = np.nanmax(np.abs(V_ramos_band[~current_fixed_mask_ramos_band] - V_old_ramos_band[~current_fixed_mask_ramos_band]))
        iteration_ramos_band += 1
        if iteration_ramos_band % 100 == 0:
            convergence_pourcentage_ramos_band = (tolerance / sigma_ramos_band) * 100 if sigma_ramos_band != 0 else 100
            print(f"  Iteration {iteration_ramos_band}, delta = {sigma_ramos_band:.2e}, pourcentage = {convergence_pourcentage_ramos_band:.4f} %")

        if sigma_ramos_band < tolerance:
            print(f"  Iteration {iteration_ramos_band}, delta = {sigma_ramos_band:.2e}, pourcentage = 100 %")
            print(f"Convergence atteinte pour la bande inférieure {j+1} en {iteration_ramos_band} itérations.")
            break

    # Calcul du champ électrique Ramos (bande individuelle)
    Ey_ramos_band, Ex_ramos_band = np.gradient(-V_ramos_band, h, h)

    # Interpolation des données Ramos (bande individuelle) sur la ROI
    V_ramos_band_interp = griddata(points_global_grid, V_ramos_band.flatten(), (X_interp, Y_interp), method='cubic')
    Ey_ramos_band_interp, Ex_ramos_band_interp = np.gradient(-V_ramos_band_interp, h_interp, h_interp)

    # --- Masquage des points hors ROI après interpolation pour Ramos ---
    V_ramos_band_interp[~mask_roi_interp] = np.nan
    Ex_ramos_band_interp[~mask_roi_interp] = np.nan
    Ey_ramos_band_interp[~mask_roi_interp] = np.nan

    # Filtrage des données 1D pour l'enregistrement
    valid_mask_1d_ramos_band = ~np.isnan(V_ramos_band_interp.flatten())
    ramos_data[f'band_{j+1}'] = {
        'V': V_ramos_band_interp.flatten()[valid_mask_1d_ramos_band],
        'Ex': Ex_ramos_band_interp.flatten()[valid_mask_1d_ramos_band],
        'Ey': Ey_ramos_band_interp.flatten()[valid_mask_1d_ramos_band],
        'X': X_interp.flatten()[valid_mask_1d_ramos_band],
        'Y': Y_interp.flatten()[valid_mask_1d_ramos_band]
    }

    # --- Affichage du potentiel et du champ électrique Ramos (bande individuelle) ---
    fig_ramos_band, axes_ramos_band = plt.subplots(1, 2, figsize=(14, 6))
    fig_ramos_band.suptitle(f"Champ de Ramos : Bande inférieure {j+1} (1V) (Région entre les plans)")

    c_ramos_band = axes_ramos_band[0].contourf(X_interp, Y_interp, V_ramos_band_interp, levels=100, cmap='plasma')
    axes_ramos_band[0].set_title("Potentiel de Ramos (V)")
    axes_ramos_band[0].set_aspect('equal')
    plt.colorbar(c_ramos_band, ax=axes_ramos_band[0], label='Potentiel (V)')
    axes_ramos_band[0].set_xlabel("x (mm)")
    axes_ramos_band[0].set_ylabel("y (mm)")
    axes_ramos_band[0].set_xlim(x_roi_min - display_padding, x_roi_max + display_padding)
    axes_ramos_band[0].set_ylim(y_roi_min - display_padding, y_roi_max + display_padding)
    axes_ramos_band[0].axhline(y_100, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    axes_ramos_band[0].axhline(y_0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)


    strm_ramos_band = axes_ramos_band[1].streamplot(X_interp, Y_interp, Ex_ramos_band_interp, Ey_ramos_band_interp,
                                                  color=np.sqrt(Ex_ramos_band_interp**2 + Ey_ramos_band_interp**2), cmap='viridis', density=1.5)
    axes_ramos_band[1].set_title("Champ Électrique de Ramos |E| (V/mm)")
    axes_ramos_band[1].set_aspect('equal')
    plt.colorbar(strm_ramos_band.lines, ax=axes_ramos_band[1], label='|E| (V/mm)')
    axes_ramos_band[1].set_xlabel("x (mm)")
    axes_ramos_band[1].set_ylabel("y (mm)")
    axes_ramos_band[1].set_xlim(x_roi_min - display_padding, x_roi_max + display_padding)
    axes_ramos_band[1].set_ylim(y_roi_min - display_padding, y_roi_max + display_padding)
    axes_ramos_band[1].axhline(y_100, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    axes_ramos_band[1].axhline(y_0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Sauvegarde des données ---
# Filtrage des données du champ principal pour l'enregistrement
valid_mask_1d_main = ~np.isnan(V_interp.flatten())

np.savez("champ_electrique_donnees_ROI.npz",
         plan_gap=plan_gap, band_width=band_width, gap=gap,
         nb_bandes=nb_bandes, R=R, total_width_bottom=total_width_bottom,
         # Données du champ principal, interpolées et limitées à la ROI (format 1D filtré)
         X_interp=X_interp.flatten()[valid_mask_1d_main],
         Y_interp=Y_interp.flatten()[valid_mask_1d_main],
         V_interp=V_interp.flatten()[valid_mask_1d_main],
         Ex_interp=Ex_interp.flatten()[valid_mask_1d_main],
         Ey_interp=Ey_interp.flatten()[valid_mask_1d_main],
         h_interp=h_interp,
         # Données des champs de Ramos, déjà interpolées et limitées à la ROI (format 1D filtré)
         ramos_data=ramos_data,
         width_100=width_100)
print("\nDonnées sauvegardées dans 'champ_electrique_donnees_ROI.npz'")
