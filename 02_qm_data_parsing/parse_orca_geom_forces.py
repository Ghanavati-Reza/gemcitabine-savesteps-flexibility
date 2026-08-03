"""
ORCA-adapted replacement for get_geom_forces_OUTCAR_AIMD_1_30_2024.py /
get_geom_forces_OUTCAR_AIMD_Hessian_1_30_2024.py.

VASP's OUTCAR TOTAL-FORCE block gives, per atom, [x, y, z (Angstrom), Fx, Fy, Fz
(eV/Angstrom)]. ORCA's .engrad files (both the static single-point Hessian-
displacement jobs and the per-step %md Dump EnGrad dumps -- confirmed identical
format by inspection of mdstep000001.engrad from the local AIMD pilot run) give
atomic numbers + coordinates in Bohr and the energy gradient in Hartree/Bohr.
This script converts each .engrad into the same [x,y,z,Fx,Fy,Fz] row layout
(Angstrom / eV*Angstrom^-1) and stacks them with np.savetxt exactly as the
original VASP parsers did, so output is a drop-in replacement for
geom_forces_array_{name}_w_Hess.csv / geom_forces_array_{name}_val.csv.

Force = -gradient (standard sign convention used by the VASP OUTCAR TOTAL-FORCE
block and consumed downstream by SAVESTEPS Step 6/7).
"""
import os
import re
import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = 27.211386245988 / BOHR_TO_ANGSTROM  # 51.42208619083232

# CRITICAL: every ORCA .engrad file (Hessian-displacement AND AIMD mdstep
# dumps alike) writes atoms in a fixed connectivity-derived order that is
# NOT the same as GEMCIT's own element-grouped numbering used throughout the
# rest of the SAVESTEPS pipeline (topology instance lists, LAMMPS build,
# etc). Confirmed via nearest-neighbor nuclear-position matching between the
# Hessian "eq" job and GEMCIT_atom_types_with_coordinates.txt: EVERY atom
# matched at exactly 0.0000 Angstrom (same physical atoms, deterministically
# reordered, not a different geometry). Without this remap, every bond
# length/angle computed downstream from engrad-derived geometries pairs the
# wrong atoms together -- confirmed by checking physical bond-length
# sanity: GEMCIT-ordering interpretation gives 30/30 sensible bond lengths,
# ORCA-native-ordering interpretation gives only 5/30 (coincidental).
# Maps 1-based ORCA .engrad atom index -> 1-based GEMCIT atom index.
ORCA_TO_GEMCIT_1BASED = {
    1: 1, 2: 4, 3: 2, 4: 5, 5: 13, 6: 3, 7: 6, 8: 7, 9: 8, 10: 14,
    11: 9, 12: 10, 13: 15, 14: 11, 15: 16, 16: 12, 17: 17, 18: 18,
    19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26,
    27: 27, 28: 28, 29: 29,
}


def _remap_to_gemcit_order(arr):
    """arr: (natoms, k) ndarray in ORCA .engrad atom order -> same shape in
    GEMCIT atom order. Only defined/validated for GEMCIT's 29 atoms; for any
    other atom count this is a no-op (so this module stays reusable)."""
    n = arr.shape[0]
    if n != 29:
        return arr
    out = np.empty_like(arr)
    for orca_idx_1based, gemcit_idx_1based in ORCA_TO_GEMCIT_1BASED.items():
        out[gemcit_idx_1based - 1] = arr[orca_idx_1based - 1]
    return out


def parse_engrad(path):
    with open(path) as f:
        lines = f.readlines()

    def find_section(tag, start=0):
        for i in range(start, len(lines)):
            if tag in lines[i]:
                return i
        raise ValueError(f"section '{tag}' not found in {path}")

    i_natoms = find_section("Number of atoms")
    n_atoms = int(lines[i_natoms + 2].strip())

    i_grad = find_section("current gradient")
    grad_start = i_grad + 2
    gradient = np.array(
        [float(lines[grad_start + i].strip()) for i in range(3 * n_atoms)]
    ).reshape(n_atoms, 3)

    i_coords = find_section("current coordinates")
    coord_start = i_coords + 2
    coords_bohr = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        parts = lines[coord_start + i].split()
        coords_bohr[i] = [float(parts[1]), float(parts[2]), float(parts[3])]

    coords_ang = coords_bohr * BOHR_TO_ANGSTROM
    forces_ev_ang = -gradient * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
    coords_ang = _remap_to_gemcit_order(coords_ang)
    forces_ev_ang = _remap_to_gemcit_order(forces_ev_ang)
    return coords_ang, forces_ev_ang


def collect_engrad_jobs(jobs_dir, name_filter=None):
    """Returns sorted list of .engrad file paths under jobs_dir (one level of
    job subdirectories each containing job.engrad, or a flat dir of *.engrad
    files e.g. mdstepNNNNNN.engrad)."""
    paths = []
    for entry in sorted(os.listdir(jobs_dir)):
        full = os.path.join(jobs_dir, entry)
        if os.path.isdir(full):
            if name_filter and not name_filter(entry):
                continue
            engrad = os.path.join(full, "job.engrad")
            status = os.path.join(full, "STATUS")
            if os.path.exists(engrad) and (
                not os.path.exists(status) or open(status).read().strip() == "0"
            ):
                paths.append(engrad)
        elif entry.endswith(".engrad"):
            paths.append(full)
    return paths


def write_geom_forces_csv(engrad_paths, out_csv_path):
    n_geom = 0
    n_atoms_ref = None
    with open(out_csv_path, "w") as out:
        for p in engrad_paths:
            coords_ang, forces_ev_ang = parse_engrad(p)
            if n_atoms_ref is None:
                n_atoms_ref = coords_ang.shape[0]
            elif coords_ang.shape[0] != n_atoms_ref:
                raise ValueError(f"{p}: atom count mismatch")
            block = np.hstack([coords_ang, forces_ev_ang])
            np.savetxt(out, block, delimiter=",")
            n_geom += 1
    print(f"{out_csv_path}: wrote {n_geom} geometries x {n_atoms_ref} atoms")
    return n_geom


def main():
    material = "GEMCIT"
    out_dir = os.path.expanduser("~/orca_runs/geom_forces_parsed")
    os.makedirs(out_dir, exist_ok=True)

    hess_dir = os.path.expanduser("~/orca_runs/hessian_disp_jobs")
    hess_paths = collect_engrad_jobs(hess_dir)
    print(f"Found {len(hess_paths)} Hessian-displacement .engrad files")

    aimd_training_dir = os.path.expanduser("~/orca_runs/aimd_training")  # 10 trajectories, populated later
    aimd_val_dir = os.path.expanduser("~/orca_runs/aimd_validation")  # 10 trajectories, populated later

    def natural_mdstep_key(p):
        m = re.search(r"mdstep(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0

    training_paths = list(hess_paths)
    if os.path.isdir(aimd_training_dir):
        for traj in sorted(os.listdir(aimd_training_dir)):
            traj_dir = os.path.join(aimd_training_dir, traj)
            if os.path.isdir(traj_dir):
                md_paths = sorted(
                    (f for f in os.listdir(traj_dir) if f.startswith("mdstep") and f.endswith(".engrad")),
                    key=lambda f: natural_mdstep_key(f),
                )
                training_paths.extend(os.path.join(traj_dir, f) for f in md_paths)
        print(f"Found AIMD training trajectories in {aimd_training_dir}")
    else:
        print(f"NOTE: {aimd_training_dir} does not exist yet -- writing Hessian-only training CSV for now")

    write_geom_forces_csv(
        training_paths, os.path.join(out_dir, f"geom_forces_array_{material}_w_Hess.csv")
    )

    val_paths = []
    if os.path.isdir(aimd_val_dir):
        for traj in sorted(os.listdir(aimd_val_dir)):
            traj_dir = os.path.join(aimd_val_dir, traj)
            if os.path.isdir(traj_dir):
                md_paths = sorted(
                    (f for f in os.listdir(traj_dir) if f.startswith("mdstep") and f.endswith(".engrad")),
                    key=lambda f: natural_mdstep_key(f),
                )
                val_paths.extend(os.path.join(traj_dir, f) for f in md_paths)
        write_geom_forces_csv(
            val_paths, os.path.join(out_dir, f"geom_forces_array_{material}_val.csv")
        )
    else:
        print(f"NOTE: {aimd_val_dir} does not exist yet -- validation CSV not written")


if __name__ == "__main__":
    main()
