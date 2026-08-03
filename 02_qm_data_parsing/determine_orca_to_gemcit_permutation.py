import sys, math, ast, json
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


def centroid(pts):
    n = len(pts)
    return tuple(sum(p[i] for p in pts) / n for i in range(3))


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
gemcit_elem = [a[0] for a in gemcit]
gemcit_xyz = [(a[1], a[2], a[3]) for a in gemcit]

eq_coords_ang, _ = parse_engrad("/home/rgh_bio/orca_runs/hessian_disp_jobs/eq/job.engrad")
eq_xyz = [tuple(row) for row in eq_coords_ang]
eq_elem = ["N", "C", "N", "C", "O", "N", "C", "C", "C", "O", "C", "C", "O", "C", "O", "C", "F", "F"] + ["H"] * 11

natoms = 29
gc = centroid(gemcit_xyz)
ec = centroid(eq_xyz)
gemcit_c = [(p[0]-gc[0], p[1]-gc[1], p[2]-gc[2]) for p in gemcit_xyz]
eq_c = [(p[0]-ec[0], p[1]-ec[1], p[2]-ec[2]) for p in eq_xyz]

# for each ORCA-order index (0-based), find nearest GEMCIT-order index (0-based) of same element
used = set()
orca_to_gemcit = [None] * natoms  # orca_to_gemcit[orca_idx_0based] = gemcit_idx_0based
match_dists = []
for oi in range(natoms):
    best_j, best_d = None, None
    for gi in range(natoms):
        if gi in used or gemcit_elem[gi] != eq_elem[oi]:
            continue
        d = dist(eq_c[oi], gemcit_c[gi])
        if best_d is None or d < best_d:
            best_d, best_j = d, gi
    used.add(best_j)
    orca_to_gemcit[oi] = best_j
    match_dists.append(best_d)

print("ORCA idx (1-based) -> GEMCIT idx (1-based), element, match distance:")
for oi in range(natoms):
    print(f"  orca[{oi+1:2d}] ({eq_elem[oi]}) -> gemcit[{orca_to_gemcit[oi]+1:2d}] ({gemcit_elem[orca_to_gemcit[oi]]})   dist={match_dists[oi]:.4f} A")

print(f"\nmax match distance: {max(match_dists):.4f} A, mean: {sum(match_dists)/len(match_dists):.4f} A")

# verify: remap eq geometry into GEMCIT order and recompute bond lengths
remapped = [None] * natoms
for oi in range(natoms):
    remapped[orca_to_gemcit[oi]] = eq_xyz[oi]

bond_instances = parse_instances(f"{BASE}/GEMCIT_bond_and_UB_instances_list.txt")
print("\nverification: bond lengths using REMAPPED eq geometry vs GEMCIT file (should now closely agree):")
max_diff = 0.0
for b in bond_instances:
    a_atom, b_atom, t = b[0], b[2], b[4]
    d1 = dist(gemcit_xyz[a_atom - 1], gemcit_xyz[b_atom - 1])
    d2 = dist(remapped[a_atom - 1], remapped[b_atom - 1])
    diff = abs(d1 - d2)
    max_diff = max(max_diff, diff)
    print(f"  type={t:2d} ({a_atom:2d},{b_atom:2d}): gemcit={d1:.4f}  remapped-eq={d2:.4f}  diff={diff:.4f}")
print(f"\nmax bond-length diff after remap: {max_diff:.4f} A")

# save the permutation (as ORCA 1-based index -> GEMCIT 1-based index) for reuse
perm_1based = {oi + 1: orca_to_gemcit[oi] + 1 for oi in range(natoms)}
with open("/home/rgh_bio/orca_to_gemcit_permutation.json", "w") as f:
    json.dump(perm_1based, f, indent=2)
print("\nsaved permutation to ~/orca_to_gemcit_permutation.json")
