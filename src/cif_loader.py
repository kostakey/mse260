from pymatgen.core import Structure
import numpy as np

def load_cif(cif_file, supercell=(2,2,2)):
    """
    Load a CIF and expand to supercell.

    Returns:
        positions (Nx3 np.array)
        atom_types (Nx1 array of integers)
        species_list (list of species strings)
    """
    structure = Structure.from_file(cif_file)
    structure = structure * supercell

    positions = structure.cart_coords
    species = [str(s) for s in structure.species]

    # Map atom types (1=cation, 2=anion)
    atom_types = []
    for s in species:
        if s == "Na+":   # example cation
            atom_types.append(1)
        elif s == "Cl-": # example anion
            atom_types.append(2)
        else:
            raise ValueError(f"Unknown species {s}")

    return np.array(positions), np.array(atom_types), species