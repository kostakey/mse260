from lammps import lammps
import numpy as np

# class LAMMPSController:
#     def __init__(self, positions, atom_types, lattice_matrix, r_anion=1.0, epsilon=1.0):
#         # Suppress LAMMPS logging
#         self.lmp = lammps(cmdargs=["-log", "none"])
#         self.atom_types = np.array(atom_types)
#         self.positions = np.array(positions)
#         self.lattice = np.array(lattice_matrix)
#         self.r_anion = r_anion
#         self.epsilon = epsilon
#         self._setup_lammps()

#     def _setup_lammps(self):
#         lx, ly, lz = self.lattice[0][0], self.lattice[1][1], self.lattice[2][2]
#         self.lmp.command("units metal")
#         self.lmp.command("atom_style atomic")
#         self.lmp.command("boundary p p p")
#         self.lmp.command(f"region box block 0 {lx} 0 {ly} 0 {lz}")
#         self.lmp.command("create_box 2 box")  # 2 atom types: cation=1, anion=2

#         # Create atoms in LAMMPS
#         for pos, typ in zip(self.positions, self.atom_types):
#             self.lmp.command(f"create_atoms {typ} single {pos[0]} {pos[1]} {pos[2]}")

#         self.lmp.command("mass 1 1.0")
#         self.lmp.command("mass 2 1.0")
#         self.lmp.command("pair_style lj/cut 2.5")
#         self.lmp.command("pair_modify shift yes")

#     def update_radius(self, r_c):
#         """Update cation radius and minimize"""
#         sigma_cc = 2 * r_c
#         sigma_aa = 2 * self.r_anion
#         sigma_ca = r_c + self.r_anion
#         self.lmp.command(f"pair_coeff 1 1 {self.epsilon} {sigma_cc}")
#         self.lmp.command(f"pair_coeff 2 2 {self.epsilon} {sigma_aa}")
#         self.lmp.command(f"pair_coeff 1 2 {self.epsilon} {sigma_ca}")
#         # Minimize without verbose output
#         self.lmp.command("minimize 1e-6 1e-6 500 5000")

#     def get_positions(self):
#         """Return positions and corresponding atom types"""
#         positions = np.array(self.lmp.numpy.extract_atom("x"))
#         types = np.array(self.lmp.numpy.extract_atom("type"))
#         return positions, types

# def run_coordination_step(positions, atom_types, type_map, cation_r, anion_r=1.0):
#     # --- 1. CENTER THE STRUCTURE ---
#     # Move the geometric center of the atoms to (0,0,0) 
#     # so scaling happens uniformly from the middle.
#     centroid = np.mean(positions, axis=0)
#     positions = positions - centroid

#     # --- 2. CALCULATE SCALE BASED ON NEAREST NEIGHBOR ---
#     # Instead of just grabbing the first pair, let's find the actual 
#     # bond length in the CIF.
#     cat_indices = np.where(atom_types == 1)[0]
#     an_indices = np.where(atom_types == 2)[0]
    
#     # Distance between the first cation and all anions
#     diff = positions[an_indices] - positions[cat_indices[0]]
#     dists = np.linalg.norm(diff, axis=1)
#     initial_bond_length = np.min(dists) # This is the CIF's 'natural' bond
    
#     # Target bond length in LJ units is cation_r + anion_r
#     scale = (cation_r + anion_r) / initial_bond_length
#     scaled_positions = positions * scale

#     # --- 3. LAMMPS SETUP (with Safety) ---
#     L = lammps(cmdargs=["-log", "none"])
#     L.command("units lj")
#     L.command("atom_style atomic")
#     L.command("boundary s s s") # Box grows to fit atoms

#     L.command("region box block -20 20 -20 20 -20 20")
#     L.command(f"create_box {len(type_map)} box")
    
#     for i, pos in enumerate(scaled_positions):
#         L.command(f"create_atoms {atom_types[i]} single {pos[0]} {pos[1]} {pos[2]}")

#     # --- 4. POTENTIALS ---
#     sig_ca = cation_r + anion_r
#     sig_aa = 2 * anion_r
#     # Use a cutoff that captures at least the first neighbor shell
#     cutoff = 1.122 * max(sig_ca, sig_aa) 
    
#     # L.command("mass * 1.0")
#     # L.command(f"pair_style lj/cut {cutoff}")
#     # L.command(f"pair_coeff 1 2 1.0 {sig_ca}") 
#     # L.command(f"pair_coeff 2 2 1.0 {sig_aa}")
#     # L.command(f"pair_coeff 1 1 1.0 {sig_ca}") # Keep cations apart too
#     # L.command("pair_modify shift yes")

#     L.command("mass * 1.0")
#     L.command(f"pair_style lj/cut {cutoff}")

#     # 1. SET A GLOBAL DEFAULT (This fixes the "not set" error)
#     # This tells LAMMPS: "By default, treat everyone like an anion"
#     L.command(f"pair_coeff * * 1.0 {sig_aa}")

#     # 2. OVERRIDE SPECIFIC INTERACTIONS
#     # Interaction between Cation (Type 1) and Anion (Type 2)
#     L.command(f"pair_coeff 1 2 1.0 {sig_ca}") 

#     # Interaction for Cation self-pairing (Type 1 with Type 1)
#     L.command(f"pair_coeff 1 1 1.0 {2*cation_r}")

#     L.command("pair_modify shift yes")

#     # --- 5. MINIMIZE ---
#     L.command("min_style fire")
#     L.command("min_modify dmax 0.05") # Safety rail
#     L.command("minimize 1.0e-6 1.0e-8 1000 10000")
    
#     coords = L.gather_atoms("x", 1, 3)
#     L.close()
#     return np.array(coords).reshape(-1, 3)

# def run_coordination_step(positions, atom_types, type_map, cation_r, anion_r=1.0):
#     L = lammps(cmdargs=["-log", "none"])
#     L.command("units lj")
#     L.command("atom_style atomic")
    
#     # 1. Use 'm' (mirror) or 's' (shrink) with a huge limit to avoid losing atoms
#     L.command("boundary m m m") 
#     L.command("neighbor 2.0 bin")
#     L.command("neigh_modify delay 0 every 1 check yes")

#     # 2. IMPROVED SCALING & CENTERING
#     centroid = np.mean(positions, axis=0)
#     centered_positions = positions - centroid

#     # Find the smallest distance in the CIF to ensure we don't overlap too much
#     # We'll use a safer scale that starts them slightly FURTHER apart
#     cat_indices = np.where(atom_types == 1)[0]
#     an_indices = np.where(atom_types == 2)[0]
    
#     # Calculate distance from first cation to its nearest anion
#     diffs = centered_positions[an_indices] - centered_positions[cat_indices[0]]
#     initial_dist = np.min(np.linalg.norm(diffs, axis=1))
    
#     # Scale such that they start exactly at the sum of radii + a 10% buffer
#     # This buffer prevents the 10^9 energy spike
#     # target_dist = (cation_r + anion_r) * 1.1 
#     # scale = target_dist / initial_dist
#     # scaled_positions = centered_positions * scale

#     # Force atoms to start exactly at the touching point
#     target_dist = (cation_r + anion_r)
#     scale = target_dist / initial_dist
#     scaled_positions = centered_positions * scale

#     # 3. CREATE BOX & ATOMS
#     L.command("region box block -100 100 -100 100 -100 100")
#     L.command(f"create_box {len(type_map)} box")
#     for i, pos in enumerate(scaled_positions):
#         L.command(f"create_atoms {atom_types[i]} single {pos[0]} {pos[1]} {pos[2]}")

#     L.command("mass * 1.0")

#     # 4. WARM-UP STEP: Use 'soft' potential to resolve overlaps
#     # This replaces the infinite push of LJ with a finite push
#     # L.command("pair_style soft 2.5")
#     # L.command("pair_coeff * * 10.0") # Max energy of 10.0 instead of 10^9
#     # L.command("minimize 1.0e-4 1.0e-6 100 1000")
    
#     L.command(f"pair_style soft {target_dist * 1.5}") # Cutoff is only 1.5x the bond
#     L.command("pair_coeff * * 5.0") # Lower energy cap (5.0 instead of 10.0)
#     L.command("minimize 1.0e-2 1.0e-3 50 500") # Fewer steps to prevent 'drifting'

#     # 5. ACTUAL PHYSICS: Lennard-Jones
#     sig_ca = cation_r + anion_r
#     sig_aa = 2 * anion_r
#     # cutoff = 1.122 * max(sig_ca, sig_aa)
    
#     # Increase the cutoff multiplier from 1.122 to 2.5
#     # This ensures atoms can 'find' each other even if they drifted slightly
#     cutoff = 2.5 * max(sig_ca, sig_aa)
#     L.command(f"pair_style lj/cut {cutoff}")

#     # L.command(f"pair_style lj/cut {cutoff}")
#     L.command(f"pair_coeff * * 1.0 {sig_aa}") 
#     L.command(f"pair_coeff 1 2 1.0 {sig_ca}") 
#     L.command("pair_modify shift yes")

#     L.command("min_style fire")
#     L.command("min_modify dmax 0.05") # Can be slightly larger now
    
#     print(f"Final relaxation for r_cat={cation_r:.3f}...")
#     L.command("minimize 1.0e-6 1.0e-8 1000 10000")
    
#     coords = L.gather_atoms("x", 1, 3)
#     final_coords = np.array(coords).reshape(-1, 3)
    
#     # Calculate the actual distance between atom 0 (Cation) and atom 1 (Anion)
#     dists = L.gather_atoms("x", 1, 3)
#     p = np.array(dists).reshape(-1, 3)
#     actual_bond = np.linalg.norm(p[0] - p[1])
#     print(f"DEBUG: Radius Sum: {cation_r + anion_r:.3f} | Actual Bond: {actual_bond:.3f}")
    
#     L.close()

    

#     return final_coords

def run_coordination_step(positions, atom_types, type_map, cation_r, anion_r=1.0):
    L = lammps(cmdargs=["-log", "none"])
    L.command("units lj")
    L.command("atom_style atomic")
    
    # 1. FIXED BOUNDARY: This is the only way to stop the 'Too many bins' crash
    L.command("boundary f f f") 
    L.command("neighbor 1.0 bin")
    L.command("neigh_modify delay 0 every 1 check yes")

    # 2. CENTER AND SCALE (No 'Target Start' buffer needed now)
    centroid = np.mean(positions, axis=0)
    centered_positions = positions - centroid
    
    cat_idx = np.where(atom_types == 1)[0][0]
    an_idx = np.where(atom_types == 2)[0][0]
    initial_dist = np.linalg.norm(centered_positions[cat_idx] - centered_positions[an_idx])
    
    # Scale exactly to the sum of radii
    scale = (cation_r + anion_r) / initial_dist
    scaled_positions = centered_positions * scale

    # 3. CREATE A GIANT FIXED BOX
    # A box from -100 to 100 is huge compared to your atoms (dist ~2.0)
    # This ensures bins stay stable.
    L.command("region box block -100 100 -100 100 -100 100")
    L.command(f"create_box {len(type_map)} box")
    for i, pos in enumerate(scaled_positions):
        L.command(f"create_atoms {atom_types[i]} single {pos[0]} {pos[1]} {pos[2]}")

    L.command("mass * 1.0")

    # 4. PHASE 1: THE SOFT PUSH (Prevents the 10^9 Explosion)
    # The 'soft' potential has a maximum energy. It can't explode.
    L.command("pair_style soft 2.5")
    L.command("pair_coeff * * 10.0") 
    L.command("minimize 1.0e-2 1.0e-3 100 1000")

    # 5. PHASE 2: REAL LENNARD-JONES
    sig_ca = (cation_r + anion_r) / 1.12246
    sig_aa = (2 * anion_r) / 1.12246
    
    L.command(f"pair_style lj/cut 5.0") # Reasonable cutoff
    L.command(f"pair_coeff * * 1.0 {sig_aa}") 
    L.command(f"pair_coeff 1 2 1.0 {sig_ca}") 
    L.command("pair_modify shift yes")

    # 6. FINAL RELAXATION
    L.command("min_style fire")
    L.command("min_modify dmax 0.02") 

    print(f"Finalizing structure for r_cat={cation_r:.3f}...")
    L.command("minimize 1.0e-6 1.0e-8 1000 10000")
    
    coords = L.gather_atoms("x", 1, 3)
    final_p = np.array(coords).reshape(-1, 3)
    L.close()
    return final_p