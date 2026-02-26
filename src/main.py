from cif_loader import load_cif
from lammps_controller import LAMMPSController
from visualization import Visualization
import numpy as np

FILEPATH = "/home/kostakey/repos/mse260/src/structures/ICSD_CollCode18189.cif"

# Load CIF
positions, atom_types, species = load_cif(FILEPATH, supercell=(2,2,2))

# Assume lattice vectors from pymatgen (approximate box)
lattice_matrix = np.eye(3) * 5.0  # Replace with actual lattice vectors if available

# Initialize LAMMPS controller
lmp_ctrl = LAMMPSController(positions, atom_types, lattice_matrix, r_anion=1.0, epsilon=1.0)

# Initialize visualization
viz = Visualization(lmp_ctrl, r_anion=1.0)
viz.show()