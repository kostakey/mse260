# from cif_loader import load_cif
# # from lammps_controller import LAMMPSController
# from lammps_controller import run_coordination_step
# from visualization import visualize_coordination
# import numpy as np
# from lammps import lammps
# import pyvista as pv

# import numpy as np
# import pyvista as pv
# from lammps import lammps
# from cif_loader import load_cif
# import matplotlib.pyplot as plt

# # config
# FILEPATH = "/home/serga/repos/mse260/src/structures/ICSD_CollCode18189.cif"
# ANION_R = 1.0 
# CATION_RADII = np.linspace(1.0, 0.1, 50)

# def check_stability(final_coords, atom_types, r_cat, r_an):
#     """Checks if the cation is 'rattling' in its cage."""
#     cat_pos = final_coords[atom_types == 1][0]
#     anion_positions = final_coords[atom_types == 2]
    
#     dists = np.linalg.norm(anion_positions - cat_pos, axis=1)
#     min_dist = np.min(dists)
    
#     ideal_dist = r_cat + r_an
#     gap = min_dist - ideal_dist
    
#     # Stable if the gap is small (overlap/contact)
#     is_stable = gap < 0.03 
#     return is_stable, gap

# def run_coordination_step(positions, atom_types, type_map, cation_r, anion_r=1.0):
#     """Standard LAMMPS minimization for a given set of atoms."""
#     L = lammps(cmdargs=["-log", "none"])
#     L.command("units lj")
#     L.command("atom_style atomic")
#     L.command("boundary f f f") 
#     L.command("neighbor 1.0 bin")
#     L.command("neigh_modify delay 0 every 1 check yes")

#     # Center and Scale
#     centroid = np.mean(positions, axis=0)
#     centered_positions = positions - centroid
    
#     # Create Box and Atoms
#     L.command("region box block -100 100 -100 100 -100 100")
#     L.command(f"create_box {len(type_map)} box")
#     for i, pos in enumerate(centered_positions):
#         L.command(f"create_atoms {atom_types[i]} single {pos[0]} {pos[1]} {pos[2]}")

#     L.command("mass * 1.0")

#     # Potentials
#     sig_ca = (cation_r + anion_r) / 1.12246
#     sig_aa = (2 * anion_r) / 1.12246
    
#     L.command("pair_style lj/cut 5.0")
#     L.command(f"pair_coeff * * 1.0 {sig_aa}") 
#     L.command(f"pair_coeff 1 2 1.0 {sig_ca}") 
#     L.command("pair_modify shift yes")

#     # Relax
#     L.command("min_style fire")
#     L.command("minimize 1.0e-6 1.0e-8 1000 10000")
    
#     coords = L.gather_atoms("x", 1, 3)
#     final_p = np.array(coords).reshape(-1, 3)
#     L.close()
#     return final_p


# positions, atom_types, type_map = load_cif(FILEPATH, supercell=(1,1,1))
# history = [] # Will store (positions, types) tuples

# current_pos = positions.copy()
# current_types = atom_types.copy()

# for r_cat in CATION_RADII:
#     print(f"\n--- Testing Radius: {r_cat:.3f} ---")
    
#     while True:
  
#         refined_pos = run_coordination_step(current_pos, current_types, type_map, r_cat, ANION_R)
 
#         stable, gap = check_stability(refined_pos, current_types, r_cat, ANION_R)
        
#         num_anions = np.sum(current_types == 2)
        
#         if stable or num_anions <= 2:
#             # Structure is happy, save and move to next radius
#             current_pos = refined_pos
#             history.append((current_pos.copy(), current_types.copy()))
#             break
#         else:
#             # Structure is unstable, remove the outlier anion
#             print(f"   Unstable (Gap: {gap:.3f}). Removing 1 anion. Remaining: {num_anions-1}")
#             cat_pos = refined_pos[current_types == 1][0]
#             anion_idx = np.where(current_types == 2)[0]
            
#             dists = np.linalg.norm(refined_pos[anion_idx] - cat_pos, axis=1)
#             furthest_anion = anion_idx[np.argmax(dists)]
            
#             current_pos = np.delete(refined_pos, furthest_anion, axis=0)
#             current_types = np.delete(current_types, furthest_anion)

# # --- VISUALIZATION ---

# plotter = pv.Plotter(window_size=[1000, 800])
# plotter.set_background("white")

# # def update_radius_real(radius_value):
# #     idx = (np.abs(CATION_RADII - radius_value)).argmin()
# #     pos, types = history[idx]
# #     r_cat_current = CATION_RADII[idx]
    
# #     # Create glyphs for current atom set
# #     points = pv.PolyData(pos)
# #     points["radius"] = np.where(types == 1, r_cat_current, ANION_R)
# #     points["atom_type"] = types
    
# #     sphere = pv.Sphere(theta_resolution=30, phi_resolution=30)
# #     glyphs = points.glyph(scale="radius", geom=sphere, factor=1.95)
    
# #     plotter.add_mesh(glyphs, scalars="atom_type", name="atoms", 
# #                      cmap=["#e74c3c", "#3498db"], show_scalar_bar=False)
    
# #     # Calculate current CN
# #     cn = np.sum(types == 2)
# #     is_stable, gap = check_stability(pos, types, r_cat_current, ANION_R)
    
# #     status_text = f"Radius: {r_cat_current:.3f} | CN: {cn}\n"
# #     status_text += "STABLE" if is_stable else f"UNSTABLE (Gap: {gap:.3f})"
# #     color = "green" if is_stable else "red"
    
# #     plotter.add_text(status_text, name="label", position='upper_left', color=color, font_size=12)

# def update_radius_real(radius_value):
#     idx = (np.abs(CATION_RADII - radius_value)).argmin()
#     pos, types = history[idx]
#     r_cat_current = CATION_RADII[idx]
    
#     points = pv.PolyData(pos)
#     points["radius"] = np.where(types == 1, r_cat_current, ANION_R)
#     points["atom_type"] = types

#     # We create a list of labels based on the current 'types' array
#     inv_map = {v: k for k, v in type_map.items()} # Flip the map: {1: 'Na', 2: 'Cl'}
#     labels = [inv_map[t] for t in types]
  
#     sphere = pv.Sphere(theta_resolution=30, phi_resolution=30)
#     glyphs = points.glyph(scale="radius", geom=sphere, factor=1.95)
    
    
#     # plotter.add_mesh(glyphs, scalars="atom_type", name="atoms", 
#     #                  cmap=["#e74c3c", "#3498db"], show_scalar_bar=False)
    
#     plotter.add_mesh(glyphs, scalars="atom_type", name="atoms", 
#                     cmap=["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6"], show_scalar_bar=False)

#     # ["royalblue", "lightcoral"]
    
#     plotter.add_point_labels(pos, labels, name="atom_labels",
#                              font_size=14, text_color="black",
#                              shape=None, shape_opacity=0,
#                              always_visible=False,
#                              render_points_as_spheres=True, 
#                              tolerance=0.1)
    
#     # plotter.add_point_labels(
#     #     pos, 
#     #     labels, 
#     #     name="atom_labels",
#     #     font_size=14, 
#     #     text_color="black",
#     #     shape=None, 
#     #     fill_opacity=0,
#     #     always_visible=False,  # <--- Change this to False
#     #     render_points_as_spheres=True,
#     #     tolerance=0.01        # Slight offset to prevent label flickering "inside" the sphere
#     # )
    
#     # 6. Update Status Text
#     cn = np.sum(types == 2)
#     is_stable, gap = check_stability(pos, types, r_cat_current, ANION_R)
#     # status_text = f"Radius: {r_cat_current:.3f} | CN: {cn}\n"
#     status_text = f"Radius: {r_cat_current:.3f}\n"
#     status_text += "STABLE" if is_stable else f"UNSTABLE (Gap: {gap:.3f})"
#     color = "green" if is_stable else "red"
    
#     plotter.add_text(status_text, name="label", position='upper_left', color=color, font_size=12)

# plotter.add_slider_widget(
#     callback=update_radius_real,
#     rng=[CATION_RADII.min(), CATION_RADII.max()],
#     value=CATION_RADII.max(),
#     title="Shrink Cation",
#     pointa=(0.6, 0.1), pointb=(0.9, 0.1),
#     style='modern'
# )

# update_radius_real(CATION_RADII.max())
# plotter.show()


# # radii = CATION_RADII
# # cns = [np.sum(h[1] == 2) for h in history] # Extract CN from history

# # plt.figure(figsize=(8, 5))
# # plt.step(radii, cns, where='post', color='teal', linewidth=2)
# # plt.title("Coordination Number vs. Cation Radius")
# # plt.xlabel("Cation Radius ($R_c$)")
# # plt.ylabel("Coordination Number (CN)")
# # plt.gca().invert_xaxis() # Shrink from right to left
# # plt.grid(True, linestyle='--', alpha=0.7)
# # plt.show()

# import numpy as np
# import pyvista as pv
# from pymatgen.core import Structure
# from lammps import lammps

# # ================= CONFIG =================
# CIF_FILE = "/home/serga/repos/mse260/src/structures/ICSD_CollCode18189.cif"
# SUPERCELL = (2, 2, 2)

# ANION_RADIUS = 1.0
# RADII = np.linspace(1.4, 0.6, 30)

# BOND_CUTOFF = 2.6


# # ================= LOAD STRUCTURE =================
# struct = Structure.from_file(CIF_FILE)
# struct = struct * SUPERCELL

# pos0 = struct.cart_coords
# species = np.array([s.symbol for s in struct.species])

# unique = sorted(set(species))
# type_map = {s: i for i, s in enumerate(unique)}
# inv_map = {v: k for k, v in type_map.items()}
# types = np.array([type_map[s] for s in species])

# CAT = 0
# AN = 1


# # ================= LAMMPS RELAXATION =================
# def relax_structure(pos, types, r_cat, r_an):

#     L = lammps(cmdargs=["-log", "none"])

#     L.command("units lj")
#     L.command("atom_style atomic")
#     L.command("boundary p p p")

#     L.command("region box block -20 20 -20 20 -20 20")
#     L.command("create_box 2 box")

#     for i, p in enumerate(pos):
#         L.command(f"create_atoms {types[i] + 1} single {p[0]} {p[1]} {p[2]}")

#     L.command("mass * 1.0")

#     sig_ca = (r_cat + r_an) / 1.12246
#     sig_aa = (2 * r_an) / 1.12246

#     L.command("pair_style lj/cut 6.0")
#     L.command(f"pair_coeff * * 1.0 {sig_aa}")
#     L.command(f"pair_coeff 1 2 1.0 {sig_ca}")

#     L.command("min_style fire")
#     L.command("minimize 1e-6 1e-8 1000 10000")

#     coords = np.array(L.gather_atoms("x", 1, 3)).reshape(-1, 3)

#     L.close()
#     return coords


# # ================= PRECOMPUTE FRAMES =================
# print("Running LAMMPS relaxations...")

# frames = []
# for r in RADII:
#     relaxed = relax_structure(pos0, types, r, ANION_RADIUS)
#     frames.append((r, relaxed.copy(), types.copy()))


# # ================= VISUALIZATION =================
# plotter = pv.Plotter()
# plotter.set_background("white")

# state = {"actors": []}


# def clear():
#     for a in state["actors"]:
#         plotter.remove_actor(a)
#     state["actors"].clear()

#     plotter.remove_actor("labels")
#     plotter.remove_actor("info")


# # ================= COORDINATION =================
# def compute_cn(pos):
#     c = pos[types == CAT]
#     a = pos[types == AN]

#     cn_vals = []
#     for ci in c:
#         d = np.linalg.norm(a - ci, axis=1)
#         cn_vals.append(np.sum(d < BOND_CUTOFF))

#     return np.mean(cn_vals)


# # ================= UPDATE FUNCTION =================
# def update(value):

#     clear()

#     idx = (np.abs(RADII - value)).argmin()
#     r, pos, t = frames[idx]

#     # ---- ATOMS ----
#     def add_atoms(x, color, scale):
#         pts = pv.PolyData(x)
#         pts["r"] = np.full(len(x), scale)

#         glyph = pts.glyph(scale="r", geom=pv.Sphere(), factor=1.0)
#         actor = plotter.add_mesh(glyph, color=color)
#         state["actors"].append(actor)

#     add_atoms(pos[t == CAT], "red", r)
#     add_atoms(pos[t == AN], "blue", ANION_RADIUS)

#     # ---- LABELS ----
#     labels = [inv_map[i] for i in t]

#     plotter.add_point_labels(
#         pos,
#         labels,
#         font_size=14,
#         text_color="black",
#         shape=None,
#         shape_opacity=0,
#         name="labels"
#     )

#     # ---- BONDS (VESTA STYLE) ----
#     c_pos = pos[t == CAT]
#     a_pos = pos[t == AN]

#     for c in c_pos:
#         d = np.linalg.norm(a_pos - c, axis=1)
#         for n in a_pos[d < BOND_CUTOFF]:
#             line = pv.Line(c, n)
#             actor = plotter.add_mesh(line, color="black", line_width=1)
#             state["actors"].append(actor)

#     # ---- TEXT INFO ----
#     cn = compute_cn(pos)

#     plotter.add_text(
#         f"Cation radius: {r:.2f}\nCN ~ {cn:.1f}",
#         font_size=14,
#         name="info"
#     )


# # ================= SLIDER =================
# plotter.add_slider_widget(
#     callback=update,
#     rng=[RADII.min(), RADII.max()],
#     value=RADII.max(),
#     title="Cation Radius",
#     style="modern"
# )

# update(RADII.max())
# plotter.show()

import numpy as np
import pyvista as pv
from pymatgen.core import Structure
from lammps import lammps
from pathlib import Path

# ================= CONFIG =================
# BASE_DIR = Path(__file__).resolve().parent
# CIF_FILE = BASE_DIR / "structures" / "ICSD_CollCode18189.cif"
CIF_FILE = "structures/ICSD_CollCode18189.cif"
SUPERCELL = (2, 2, 2)

CATION_RADII = np.linspace(1.4, 0.6, 40)
ANION_RADIUS = 1.0

BOND_CUTOFF = 2.6


# ================= LOAD STRUCTURE =================
struct = Structure.from_file(CIF_FILE)
struct = struct * SUPERCELL

pos0 = struct.cart_coords
species = np.array([s.symbol for s in struct.species])

unique = sorted(set(species))
type_map = {s: i for i, s in enumerate(unique)}
inv_map = {v: k for k, v in type_map.items()}

types = np.array([type_map[s] for s in species])

# Safer assumption: first = cation, second = anion
CAT = 0
AN = 1


# ================= LAMMPS RELAXATION =================
def relax(pos, types, r_cat, r_an):

    L = lammps(cmdargs=["-log", "none"])

    L.command("units lj")
    L.command("atom_style atomic")
    L.command("boundary p p p")

    L.command("region box block -20 20 -20 20 -20 20")
    L.command("create_box 2 box")

    for i, p in enumerate(pos):
        L.command(f"create_atoms {types[i] + 1} single {p[0]} {p[1]} {p[2]}")

    L.command("mass * 1.0")

    sig_ca = (r_cat + r_an) / 1.12246
    sig_aa = (2 * r_an) / 1.12246

    L.command("pair_style lj/cut 6.0")
    L.command(f"pair_coeff * * 1.0 {sig_aa}")
    L.command(f"pair_coeff 1 2 1.0 {sig_ca}")

    L.command("min_style fire")
    L.command("minimize 1e-6 1e-8 1000 10000")

    coords = np.array(L.gather_atoms("x", 1, 3)).reshape(-1, 3)

    L.close()
    return coords


# ================= PRECOMPUTE FRAMES =================
print("Running LAMMPS...")
frames = []

for r in CATION_RADII:
    relaxed = relax(pos0, types, r, ANION_RADIUS)
    frames.append((r, relaxed.copy(), types.copy()))


# ================= VISUALIZATION =================
plotter = pv.Plotter()
plotter.set_background("white")

state = {"actors": []}


def clear():
    for a in state["actors"]:
        plotter.remove_actor(a)
    state["actors"].clear()

    plotter.remove_actor("labels")
    plotter.remove_actor("text")


# ================= ATOM RENDERING =================
def add_atoms(x, color, radius):
    pts = pv.PolyData(x)

    # IMPORTANT: explicit scaling per atom type
    pts["scale"] = np.full(len(x), radius)

    glyph = pts.glyph(
        scale="scale",
        geom=pv.Sphere(),
        factor=1.0,
        orient=False
    )

    actor = plotter.add_mesh(glyph, color=color)
    state["actors"].append(actor)


# ================= UPDATE FUNCTION =================
def update(value):

    clear()

    idx = (np.abs(CATION_RADII - value)).argmin()
    r, pos, t = frames[idx]

    # ---- ATOMS ----
    cat_pos = pos[t == CAT]
    an_pos = pos[t == AN]

    add_atoms(cat_pos, "red", r)              # CATION (changes)
    add_atoms(an_pos, "blue", ANION_RADIUS)   # ANION (fixed)

    # ---- LABELS ----
    labels = [inv_map[i] for i in t]

    plotter.add_point_labels(
        pos,
        labels,
        font_size=14,
        text_color="black",
        shape=None,
        shape_opacity=0,
        name="labels"
    )

    # ---- BONDS ----
    for c in cat_pos:
        d = np.linalg.norm(an_pos - c, axis=1)
        for n in an_pos[d < BOND_CUTOFF]:
            line = pv.Line(c, n)
            a = plotter.add_mesh(line, color="black", line_width=1)
            state["actors"].append(a)

    # ---- TEXT ----
    cn = np.mean([
        np.sum(np.linalg.norm(an_pos - c, axis=1) < BOND_CUTOFF)
        for c in cat_pos
    ])

    plotter.add_text(
        f"Cation radius: {r:.2f}\nAvg CN: {cn:.2f}",
        font_size=14,
        name="text"
    )


# ================= SLIDER =================
plotter.add_slider_widget(
    update,
    rng=[CATION_RADII.min(), CATION_RADII.max()],
    value=CATION_RADII.max(),
    title="Cation Radius",
    style="modern"
)

update(CATION_RADII.max())
plotter.show()

