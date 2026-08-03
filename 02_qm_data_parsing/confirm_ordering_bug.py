import sys, math, ast
sys.path.insert(0, '/home/rgh_bio/orca_runs')
from parse_orca_geom_forces import parse_engrad

BASE = "/home/rgh_bio/savesteps_run/gemcitabine/Flexibility_parameters_calculations_final_input_files/GEMCIT/Average/geometry_and_typing_files"
ELEMENT_BY_NUM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def read_gemcit_coords(path):
    with open(path) as f:
        lines = f.readlines()
    atoms = []
    for line in lines[6:6 + 29]:
        parts = line.split()
        elem = ELEMENT_BY_NUM[int(parts[4].split("[")[0])]
        atoms.append((elem, float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def parse_instances(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(ast.literal_eval(line))
    return out


gemcit = read_gemcit_coords(f"{BASE}/GEMCIT_atom_types_with_coordinates.txt")
gemcit_xyz = [(a[1], a[2], a[3]) for a in gemcit]
gemcit_elem = [a[0] for a in gemcit]

eq_coords_ang, _ = parse_engrad("/home/rgh_bio/orca_runs/hessian_disp_jobs/eq/job.engrad")
eq_xyz = [tuple(row) for row in eq_coords_ang]
eq_elem = ["N", "C", "N", "C", "O", "N", "C", "C", "C", "O", "C", "C", "O", "C", "O", "C", "F", "F"] + ["H"] * 11

bond_instances = parse_instances(f"{BASE}/GEMCIT_bond_and_UB_instances_list.txt")

print("Hypothesis A: topology indices refer to GEMCIT_atom_types_with_coordinates.txt ordering")
print("Hypothesis B: topology indices refer to eq-job.engrad (ORCA) ordering")
print()
print(f"{'type':>4} {'atoms':>8} {'elemA':>7} {'lenA':>7}   {'elemB':>7} {'lenB':>7}")
for b in bond_instances:
    a_atom, b_atom, t = b[0], b[2], b[4]
    dA = dist(gemcit_xyz[a_atom - 1], gemcit_xyz[b_atom - 1])
    dB = dist(eq_xyz[a_atom - 1], eq_xyz[b_atom - 1])
    elemA = f"{gemcit_elem[a_atom-1]}-{gemcit_elem[b_atom-1]}"
    elemB = f"{eq_elem[a_atom-1]}-{eq_elem[b_atom-1]}"
    flagA = "" if 0.8 < dA < 1.8 else "  <-- NOT a bond length"
    flagB = "" if 0.8 < dB < 1.8 else "  <-- NOT a bond length"
    print(f"{t:4d} ({a_atom:2d},{b_atom:2d})  {elemA:>7} {dA:7.4f}{flagA}   {elemB:>7} {dB:7.4f}{flagB}")

nA_ok = sum(1 for b in bond_instances if 0.8 < dist(gemcit_xyz[b[0]-1], gemcit_xyz[b[2]-1]) < 1.8)
nB_ok = sum(1 for b in bond_instances if 0.8 < dist(eq_xyz[b[0]-1], eq_xyz[b[2]-1]) < 1.8)
print(f"\nHypothesis A (GEMCIT ordering): {nA_ok}/{len(bond_instances)} bonds have a physically sensible length")
print(f"Hypothesis B (ORCA/eq ordering): {nB_ok}/{len(bond_instances)} bonds have a physically sensible length")
