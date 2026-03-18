import pyvista as pv
from scipy.spatial import cKDTree
import numpy as np

# class Visualization:
#     def __init__(self, lmp_controller, r_anion):
#         self.lmp_controller = lmp_controller
#         self.r_anion = r_anion

#         self.plotter = pv.Plotter()
#         self.positions, self.atom_types = self.lmp_controller.get_positions()

#         # Separate cations and anions
#         self.cation_positions = self.positions[self.atom_types == 1]
#         self.anion_positions  = self.positions[self.atom_types == 2]

#         # Glyphs for visualization
#         self.cation_cloud = pv.PolyData(self.cation_positions)
#         self.cation_glyph = self.cation_cloud.glyph(scale=False, geom=pv.Sphere(radius=0.3))
#         self.cation_actor = self.plotter.add_mesh(self.cation_glyph, color="red")

#         self.anion_cloud = pv.PolyData(self.anion_positions)
#         self.anion_glyph = self.anion_cloud.glyph(scale=False, geom=pv.Sphere(radius=0.3))
#         self.anion_actor = self.plotter.add_mesh(self.anion_glyph, color="blue")

#         self.text_actor = self.plotter.add_text("", font_size=14)

#     def compute_coordination(self, positions, r_c):
#         cutoff = r_c + self.r_anion + 0.1
#         tree = cKDTree(positions)
#         counts = []
#         for i, pos in enumerate(positions):
#             if self.atom_types[i] == 1:  # cation
#                 neighbors = tree.query_ball_point(pos, cutoff)
#                 counts.append(len(neighbors) - 1)
#         return np.mean(counts) if counts else 0

#     def update(self, value):
#         r_c = value * self.r_anion
#         self.lmp_controller.update_radius(r_c)
#         self.positions, self.atom_types = self.lmp_controller.get_positions()

#         # Update cation/anions separately
#         self.cation_positions = self.positions[self.atom_types == 1]
#         self.anion_positions  = self.positions[self.atom_types == 2]

#         self.cation_cloud.points = self.cation_positions
#         self.anion_cloud.points  = self.anion_positions

#         cn = self.compute_coordination(self.positions, r_c)
#         self.text_actor.SetText(0, f"Radius Ratio: {value:.2f}")
#         self.text_actor.SetText(1, f"Avg Coordination: {cn:.2f}")
#         self.plotter.render()

#     def show(self):
#         self.plotter.add_slider_widget(
#             self.update,
#             rng=[0.2, 1.2],
#             value=1.0,
#             title="Radius Ratio (r_c / r_a)"
#         )
#         self.plotter.show()

def visualize_coordination(positions, atom_types, cation_r, anion_r=1.0):
    """
    Visualizes the cation and anions using PyVista spheres.
    """
    # 1. Create a Point Cloud from coordinates
    point_cloud = pv.PolyData(positions)
    
    # 2. Add properties to the points
    point_cloud["atom_type"] = atom_types
    
    # Map radii based on type: Type 1 (Cation) = cation_r, Type 2 (Anion) = anion_r
    radii = np.where(atom_types == 1, cation_r, anion_r)
    point_cloud["radius"] = radii

    # 3. Create the Spheres (Glyphs)
    # We use a sphere source and scale it by our radius array
    sphere = pv.Sphere(theta_resolution=20, phi_resolution=20)
    glyphs = point_cloud.glyph(scale="radius", geom=sphere, factor=1.0)

    # 4. Set up the Plotter
    plotter = pv.Plotter(window_size=[1000, 1000])
    plotter.set_background("white")
    
    # Add the spheres to the scene
    # cmap defines colors: index 1 (cation) vs index 2 (anion)
    plotter.add_mesh(glyphs, scalars="atom_type", cmap=["royalblue", "lightcoral"], 
                     show_scalar_bar=False, smooth_shading=True)

    # 5. Add a bounding box for context
    plotter.add_bounding_box(color="grey")
    
    print("Opening PyVista window...")
    plotter.show()