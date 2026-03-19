# from cif_loader import load_cif
# # from lammps_controller import LAMMPSController
# from lammps_controller import run_coordination_step
# from visualization import visualize_coordination
# import numpy as np
# from lammps import lammps
# import pyvista as pv

import numpy as np
import pyvista as pv
from lammps import lammps
from cif_loader import load_cif
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
FILEPATH = "/home/serga/repos/mse260/src/structures/ICSD_CollCode18189.cif"
ANION_R = 1.0 
CATION_RADII = np.linspace(1.0, 0.1, 50)

# --- CORE FUNCTIONS ---

def check_stability(final_coords, atom_types, r_cat, r_an):
    """Checks if the cation is 'rattling' in its cage."""
    cat_pos = final_coords[atom_types == 1][0]
    anion_positions = final_coords[atom_types == 2]
    
    dists = np.linalg.norm(anion_positions - cat_pos, axis=1)
    min_dist = np.min(dists)
    
    ideal_dist = r_cat + r_an
    gap = min_dist - ideal_dist
    
    # Stable if the gap is small (overlap/contact)
    is_stable = gap < 0.03 
    return is_stable, gap

def run_coordination_step(positions, atom_types, type_map, cation_r, anion_r=1.0):
    """Standard LAMMPS minimization for a given set of atoms."""
    L = lammps(cmdargs=["-log", "none"])
    L.command("units lj")
    L.command("atom_style atomic")
    L.command("boundary f f f") 
    L.command("neighbor 1.0 bin")
    L.command("neigh_modify delay 0 every 1 check yes")

    # Center and Scale
    centroid = np.mean(positions, axis=0)
    centered_positions = positions - centroid
    
    # Create Box and Atoms
    L.command("region box block -100 100 -100 100 -100 100")
    L.command(f"create_box {len(type_map)} box")
    for i, pos in enumerate(centered_positions):
        L.command(f"create_atoms {atom_types[i]} single {pos[0]} {pos[1]} {pos[2]}")

    L.command("mass * 1.0")

    # Potentials
    sig_ca = (cation_r + anion_r) / 1.12246
    sig_aa = (2 * anion_r) / 1.12246
    
    L.command("pair_style lj/cut 5.0")
    L.command(f"pair_coeff * * 1.0 {sig_aa}") 
    L.command(f"pair_coeff 1 2 1.0 {sig_ca}") 
    L.command("pair_modify shift yes")

    # Relax
    L.command("min_style fire")
    L.command("minimize 1.0e-6 1.0e-8 1000 10000")
    
    coords = L.gather_atoms("x", 1, 3)
    final_p = np.array(coords).reshape(-1, 3)
    L.close()
    return final_p

# --- MAIN SIMULATION LOOP ---

positions, atom_types, type_map = load_cif(FILEPATH, supercell=(1,1,1))
history = [] # Will store (positions, types) tuples

current_pos = positions.copy()
current_types = atom_types.copy()

for r_cat in CATION_RADII:
    print(f"\n--- Testing Radius: {r_cat:.3f} ---")
    
    while True:
        # Step 1: Relax current geometry
        refined_pos = run_coordination_step(current_pos, current_types, type_map, r_cat, ANION_R)
        
        # Step 2: Check stability
        stable, gap = check_stability(refined_pos, current_types, r_cat, ANION_R)
        
        # Step 3: Decision - Keep or Remove?
        num_anions = np.sum(current_types == 2)
        
        if stable or num_anions <= 2:
            # Structure is happy, save and move to next radius
            current_pos = refined_pos
            history.append((current_pos.copy(), current_types.copy()))
            break
        else:
            # Structure is unstable, remove the outlier anion
            print(f"   Unstable (Gap: {gap:.3f}). Removing 1 anion. Remaining: {num_anions-1}")
            cat_pos = refined_pos[current_types == 1][0]
            anion_idx = np.where(current_types == 2)[0]
            
            dists = np.linalg.norm(refined_pos[anion_idx] - cat_pos, axis=1)
            furthest_anion = anion_idx[np.argmax(dists)]
            
            current_pos = np.delete(refined_pos, furthest_anion, axis=0)
            current_types = np.delete(current_types, furthest_anion)

# --- VISUALIZATION ---

plotter = pv.Plotter(window_size=[1000, 800])
plotter.set_background("white")

# def update_radius_real(radius_value):
#     idx = (np.abs(CATION_RADII - radius_value)).argmin()
#     pos, types = history[idx]
#     r_cat_current = CATION_RADII[idx]
    
#     # Create glyphs for current atom set
#     points = pv.PolyData(pos)
#     points["radius"] = np.where(types == 1, r_cat_current, ANION_R)
#     points["atom_type"] = types
    
#     sphere = pv.Sphere(theta_resolution=30, phi_resolution=30)
#     glyphs = points.glyph(scale="radius", geom=sphere, factor=1.95)
    
#     plotter.add_mesh(glyphs, scalars="atom_type", name="atoms", 
#                      cmap=["#e74c3c", "#3498db"], show_scalar_bar=False)
    
#     # Calculate current CN
#     cn = np.sum(types == 2)
#     is_stable, gap = check_stability(pos, types, r_cat_current, ANION_R)
    
#     status_text = f"Radius: {r_cat_current:.3f} | CN: {cn}\n"
#     status_text += "STABLE" if is_stable else f"UNSTABLE (Gap: {gap:.3f})"
#     color = "green" if is_stable else "red"
    
#     plotter.add_text(status_text, name="label", position='upper_left', color=color, font_size=12)

def update_radius_real(radius_value):
    idx = (np.abs(CATION_RADII - radius_value)).argmin()
    pos, types = history[idx]
    r_cat_current = CATION_RADII[idx]
    
    # 1. Create the PolyData
    points = pv.PolyData(pos)
    points["radius"] = np.where(types == 1, r_cat_current, ANION_R)
    points["atom_type"] = types
    
    # 2. Map types to Chemical Symbols (e.g., Type 1 -> 'Na', Type 2 -> 'Cl')
    # We create a list of labels based on the current 'types' array
    inv_map = {v: k for k, v in type_map.items()} # Flip the map: {1: 'Na', 2: 'Cl'}
    labels = [inv_map[t] for t in types]
    
    # 3. Create Glyphs
    sphere = pv.Sphere(theta_resolution=30, phi_resolution=30)
    glyphs = points.glyph(scale="radius", geom=sphere, factor=1.95)
    
    # 4. Update Mesh
    # plotter.add_mesh(glyphs, scalars="atom_type", name="atoms", 
    #                  cmap=["#e74c3c", "#3498db"], show_scalar_bar=False)
    
    plotter.add_mesh(glyphs, scalars="atom_type", name="atoms", 
                    cmap=["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6"], show_scalar_bar=False)

    # ["royalblue", "lightcoral"]
    
    # 5. Add Labels (Floating text over atoms)
    plotter.add_point_labels(pos, labels, name="atom_labels",
                             font_size=14, text_color="black",
                             shape=None, shape_opacity=0,
                             always_visible=False,
                             render_points_as_spheres=True, 
                             tolerance=0.1)
    
    # plotter.add_point_labels(
    #     pos, 
    #     labels, 
    #     name="atom_labels",
    #     font_size=14, 
    #     text_color="black",
    #     shape=None, 
    #     fill_opacity=0,
    #     always_visible=False,  # <--- Change this to False
    #     render_points_as_spheres=True,
    #     tolerance=0.01        # Slight offset to prevent label flickering "inside" the sphere
    # )
    
    # 6. Update Status Text
    cn = np.sum(types == 2)
    is_stable, gap = check_stability(pos, types, r_cat_current, ANION_R)
    status_text = f"Radius: {r_cat_current:.3f} | CN: {cn}\n"
    status_text += "STABLE" if is_stable else f"UNSTABLE (Gap: {gap:.3f})"
    color = "green" if is_stable else "red"
    
    plotter.add_text(status_text, name="label", position='upper_left', color=color, font_size=12)

plotter.add_slider_widget(
    callback=update_radius_real,
    rng=[CATION_RADII.min(), CATION_RADII.max()],
    value=CATION_RADII.max(),
    title="Shrink Cation",
    pointa=(0.6, 0.1), pointb=(0.9, 0.1),
    style='modern'
)

update_radius_real(CATION_RADII.max())
plotter.show()


# radii = CATION_RADII
# cns = [np.sum(h[1] == 2) for h in history] # Extract CN from history

# plt.figure(figsize=(8, 5))
# plt.step(radii, cns, where='post', color='teal', linewidth=2)
# plt.title("Coordination Number vs. Cation Radius")
# plt.xlabel("Cation Radius ($R_c$)")
# plt.ylabel("Coordination Number (CN)")
# plt.gca().invert_xaxis() # Shrink from right to left
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()



"""
FILEPATH = "/home/serga/repos/mse260/src/structures/ICSD_CollCode18189.cif"
ANION_R = 1.0  # Reference radius
# CATION_RADII = [1.0, 0.732, 0.414, 0.225] # Pauling's critical limits
# CATION_RADII = np.linspace(1.0, 0.2, 20)
CATION_RADII = np.linspace(1.0, 0.1, 50)

# Load CIF
positions, atom_types, type_map = load_cif(FILEPATH, supercell=(1,1,1))

# print("positions: ", positions)
# print("atom types: ", atom_types)
# print("species: ", type_map)

history = []

for r_cat in CATION_RADII:
    print(f"\n--- Simulating Cation Radius: {r_cat} ---")
    
    # Run the LAMMPS minimization for this specific radius
    # The 'run_coordination_step' function now handles scaling internally
    new_coords = run_coordination_step(
        positions, 
        atom_types, 
        type_map, 
        cation_r=r_cat, 
        anion_r=ANION_R
    )
    
    history.append(new_coords)
    print(f"Minimization complete. Atoms moved to stable equilibrium.")

# --- 2. INTERACTIVE VISUALIZATION ---
plotter = pv.Plotter(window_size=[1000, 800])
plotter.set_background("white")

# # This is the function the slider calls whenever it moves
# def update_radius(value):
#     # Map the slider value (0 to 19) to the nearest index in our history
#     idx = int(value)
#     current_pos = history[idx]
#     current_r_cat = CATION_RADII[idx]
    
#     # Update points and radii
#     points = pv.PolyData(current_pos)
#     points["radius"] = np.where(atom_types == 1, current_r_cat, ANION_R)
#     points["atom_type"] = atom_types
    
#     # Re-glyph
#     sphere = pv.Sphere(theta_resolution=20, phi_resolution=20)
#     glyphs = points.glyph(scale="radius", geom=sphere, factor=1.0)
    
#     # Update the mesh in the plotter
#     plotter.add_mesh(glyphs, scalars="atom_type", name="atoms", 
#                      cmap=["royalblue", "lightcoral"], show_scalar_bar=False)
    
#     # Update the label
#     # plotter.add_text(f"Cation Radius: {current_r_cat:.3f}", name="label", 
#     #                  position='upper_left', color="black", font_size=12)

# # Add the slider widget
# plotter.add_slider_widget(
#     callback=update_radius,
#     rng=[0, len(history) - 1],
#     value=0,
#     title="Shrink Cation",
#     pointa=(0.1, 0.1), 
#     pointb=(0.4, 0.1),
#     style='modern'
# )

# # Initialize the first frame
# update_radius(0)

# print("Opening Interactive Slider...")
# plotter.show()

def check_stability(final_coords, atom_types, r_cat, r_an):
    # 1. Identify Cation and Anions
    cat_pos = final_coords[atom_types == 1][0] # Assuming 1st cation
    anion_positions = final_coords[atom_types == 2]
    
    # 2. Calculate actual distances
    dists = np.linalg.norm(anion_positions - cat_pos, axis=1)
    min_dist = np.min(dists)
    
    # 3. The "Stability Gap"
    # Ideal distance is r_cat + r_an
    # If actual distance is significantly larger, the cation is 'rattling'
    ideal_dist = r_cat + r_an
    gap = min_dist - ideal_dist
    
    is_stable = gap < 0.05 # Tolerance for LJ potential overlap
    return is_stable, gap

def update_radius_real(radius_value):
    # Find the index of the radius in CATION_RADII closest to the slider value
    idx = (np.abs(CATION_RADII - radius_value)).argmin()
    
    current_pos = history[idx]
    current_r_cat = CATION_RADII[idx]
    
    # --- A. REAL ATOMS ---
    points = pv.PolyData(current_pos)
    points["radius"] = np.where(atom_types == 1, current_r_cat, ANION_R)
    points["atom_type"] = atom_types
    
    sphere_src = pv.Sphere(theta_resolution=30, phi_resolution=30)
    glyphs = points.glyph(scale="radius", geom=sphere_src, factor=1.95)
    
    plotter.add_mesh(glyphs, scalars="atom_type", name="atoms", 
                     cmap=["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6"], show_scalar_bar=False)

    # ["royalblue", "lightcoral"]
    
    plotter.add_text(f"Current Cation Radius: {current_r_cat:.3f}", 
                     name="label", position='upper_left', color="black", font_size=14)
    
    # # Stability warning
    # if current_r_cat < 0.414: # Example for Octahedral limit
    #     plotter.add_text("STATUS: UNSTABLE (Rattle)", name="warn", 
    #                      position='upper_right', color="red", font_size=12)
    # else:
    #     plotter.add_text("STATUS: STABLE", name="warn", 
    #                      position='upper_right', color="green", font_size=12)

    # Inside your update_radius function:
    is_stable, gap = check_stability(current_pos, atom_types, current_r_cat, ANION_R)

    status_color = "green" if is_stable else "red"
    status_text = "STABLE (Contact)" if is_stable else f"UNSTABLE (Gap: {gap:.3f})"

    plotter.add_text(status_text, name="stability_label", position="upper_right", color=status_color)
        
# Add the slider widget using REAL radius values
plotter.add_slider_widget(
    callback=update_radius_real,
    rng=[CATION_RADII.min(), CATION_RADII.max()],
    value=CATION_RADII.max(), # Start at 1.0
    title="Cation Radius (Rc)",
    pointa=(0.6, 0.1), 
    pointb=(0.9, 0.1),
    style='modern'
)

# Initialize
update_radius_real(1.0)
plotter.show()

# visualize_coordination(
#         positions=history[-1], 
#         atom_types=atom_types, 
#         cation_r=CATION_RADII[-1], 
#         anion_r=ANION_R
#     )
# Assume lattice vectors from pymatgen (approximate box)
# lattice_matrix = np.eye(3) * 5.0  # Replace with actual lattice vectors if available
"""