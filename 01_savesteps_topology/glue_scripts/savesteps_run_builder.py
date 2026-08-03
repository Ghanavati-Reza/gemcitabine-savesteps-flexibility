"""
Build a SAVESTEPS-compatible POSCAR for gemcitabine (an isolated, non-periodic
molecule) from the ORCA-optimized Cartesian xyz, using a huge dummy cubic box
so periodic images never come into play. Also writes an atom-index mapping
back to the original ORCA/xyz atom ordering, since POSCAR requires atoms
grouped by element (our xyz has them in molecule-connectivity order).
"""
import os

XYZ_IN = os.path.expanduser("~/orca_runs/gemcitabine_pilot/gemcitabine.xyz")
OUT_DIR = os.path.expanduser("~/savesteps_run/gemcitabine/converged_geo_input_flexibility_param/GEMCIT")
BOX_LENGTH = 200.0  # Angstrom, dummy cubic box -- molecule extent is ~10-12 A

def read_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    natoms = int(lines[0].split()[0])
    atoms = []
    for line in lines[2:2+natoms]:
        parts = line.split()
        atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms

def main():
    atoms = read_xyz(XYZ_IN)
    natoms = len(atoms)

    # group by element, preserving order of first appearance; record mapping back to original index
    element_order = []
    groups = {}
    for orig_idx, (sym, x, y, z) in enumerate(atoms):
        if sym not in groups:
            groups[sym] = []
            element_order.append(sym)
        groups[sym].append((orig_idx, x, y, z))

    ordered = []
    for sym in element_order:
        ordered.extend(groups[sym])
    counts = [len(groups[sym]) for sym in element_order]

    # centroid shift so all coords land comfortably inside [0, BOX_LENGTH)
    cx = sum(o[1] for o in ordered) / natoms
    cy = sum(o[2] for o in ordered) / natoms
    cz = sum(o[3] for o in ordered) / natoms
    shift = BOX_LENGTH / 2.0

    os.makedirs(OUT_DIR, exist_ok=True)

    poscar_path = os.path.join(OUT_DIR, "POSCAR")
    with open(poscar_path, "w") as f:
        f.write("GEMCIT gas-phase molecule, dummy 200A cubic box (non-periodic)\n")
        f.write("   1.00000000000000\n")
        f.write(f"   {BOX_LENGTH:.10f}    0.0000000000    0.0000000000\n")
        f.write(f"    0.0000000000   {BOX_LENGTH:.10f}    0.0000000000\n")
        f.write(f"    0.0000000000    0.0000000000   {BOX_LENGTH:.10f}\n")
        f.write("   " + "  ".join(element_order) + "\n")
        f.write("   " + "  ".join(str(c) for c in counts) + "\n")
        f.write("Selective dynamics\n")
        f.write("Direct\n")
        for orig_idx, x, y, z in ordered:
            fx = (x - cx + shift) / BOX_LENGTH
            fy = (y - cy + shift) / BOX_LENGTH
            fz = (z - cz + shift) / BOX_LENGTH
            f.write(f"  {fx:.16f}  {fy:.16f}  {fz:.16f}   T   T   T\n")

    # atom-index mapping: POSCAR order (1-based) -> original ORCA xyz order (0-based)
    map_path = os.path.join(OUT_DIR, "atom_index_map_POSCARorder_to_origXYZindex.csv")
    with open(map_path, "w") as f:
        f.write("poscar_index_1based,orig_xyz_index_0based,element\n")
        for i, (orig_idx, x, y, z) in enumerate(ordered, start=1):
            f.write(f"{i},{orig_idx},{atoms[orig_idx][0]}\n")

    print(f"Wrote POSCAR with {natoms} atoms, elements grouped as {list(zip(element_order, counts))}")
    print(f"POSCAR: {poscar_path}")
    print(f"Atom index map: {map_path}")

if __name__ == "__main__":
    main()
