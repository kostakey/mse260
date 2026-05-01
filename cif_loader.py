# from pymatgen.core import Structure
# import numpy as np

# def load_cif(cif_file, supercell=(1,1,1)):
    
#     structure = Structure.from_file(cif_file)
#     structure = structure * supercell # replicate the unit cell twice in the x, y, and z directions to avoid self-interaction errors

#     positions = structure.cart_coords # extract cartesian coords
    
#     # Find unique species and sort them by electronegativity
#     unique_species = sorted(list(set(structure.species)), key=lambda x: x.X)

#     type_map = {spec.symbol: (i + 1) for i, spec in enumerate(unique_species)}
#     atom_types = [type_map[s.symbol] for s in structure.species]

#     return np.array(positions), np.array(atom_types), type_map

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np


def load_cif(cif_file, supercell=(1,1,1)):

    structure = Structure.from_file(cif_file)

    # ================= IMPORTANT FIX =================
    # convert to conventional standard cell (VESTA-like)
    sga = SpacegroupAnalyzer(structure)
    structure = sga.get_conventional_standard_structure()

    # now apply supercell
    structure = structure * supercell

    positions = structure.cart_coords

    # species mapping
    unique_species = sorted(
        list(set(structure.species)),
        key=lambda x: x.X
    )

    type_map = {
        spec.symbol: i + 1
        for i, spec in enumerate(unique_species)
    }

    atom_types = np.array([type_map[s.symbol] for s in structure.species])

    return np.array(positions), atom_types, type_map