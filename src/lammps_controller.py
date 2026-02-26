from lammps import lammps
import numpy as np

class LAMMPSController:
    def __init__(self, positions, atom_types, lattice_matrix, r_anion=1.0, epsilon=1.0):
        # Suppress LAMMPS logging
        self.lmp = lammps(cmdargs=["-log", "none"])
        self.atom_types = np.array(atom_types)
        self.positions = np.array(positions)
        self.lattice = np.array(lattice_matrix)
        self.r_anion = r_anion
        self.epsilon = epsilon
        self._setup_lammps()

    def _setup_lammps(self):
        lx, ly, lz = self.lattice[0][0], self.lattice[1][1], self.lattice[2][2]
        self.lmp.command("units metal")
        self.lmp.command("atom_style atomic")
        self.lmp.command("boundary p p p")
        self.lmp.command(f"region box block 0 {lx} 0 {ly} 0 {lz}")
        self.lmp.command("create_box 2 box")  # 2 atom types: cation=1, anion=2

        # Create atoms in LAMMPS
        for pos, typ in zip(self.positions, self.atom_types):
            self.lmp.command(f"create_atoms {typ} single {pos[0]} {pos[1]} {pos[2]}")

        self.lmp.command("mass 1 1.0")
        self.lmp.command("mass 2 1.0")
        self.lmp.command("pair_style lj/cut 2.5")
        self.lmp.command("pair_modify shift yes")

    def update_radius(self, r_c):
        """Update cation radius and minimize"""
        sigma_cc = 2 * r_c
        sigma_aa = 2 * self.r_anion
        sigma_ca = r_c + self.r_anion
        self.lmp.command(f"pair_coeff 1 1 {self.epsilon} {sigma_cc}")
        self.lmp.command(f"pair_coeff 2 2 {self.epsilon} {sigma_aa}")
        self.lmp.command(f"pair_coeff 1 2 {self.epsilon} {sigma_ca}")
        # Minimize without verbose output
        self.lmp.command("minimize 1e-6 1e-6 500 5000")

    def get_positions(self):
        """Return positions and corresponding atom types"""
        positions = np.array(self.lmp.numpy.extract_atom("x"))
        types = np.array(self.lmp.numpy.extract_atom("type"))
        return positions, types