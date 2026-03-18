from cif_loader import load_cif
# from lammps_controller import LAMMPSController
from lammps_controller import run_coordination_step
from visualization import visualize_coordination
import numpy as np
from lammps import lammps
import pyvista as pv

FILEPATH = "/home/kostakey/repos/mse260/src/structures/ICSD_CollCode15489.cif"
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