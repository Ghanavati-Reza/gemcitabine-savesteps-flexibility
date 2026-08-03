"""
Combine the per-rotatable-dihedral-type SPDFT geoms/energy CSVs (from
parse_orca_spdft.py) into the single combined GEMCIT_geoms_SPDFT.csv /
GEMCIT_E_SPDFT.csv files Step 7's LASSO fitting script actually expects
(Materials_average_equil_flexibility_parameters_calc_*.py, ~line 1505-1507:
path8_geom_array = base_dir + Material_name + '_geoms_SPDFT.csv').

Format: 1 equilibrium geometry (29 atoms) prepended, then each RD type's 36
geometries in RD_type_ind_SPDFT.txt's order. RD_type_info_final.txt confirmed
that order is global dihedral-type index [0, 11, 13], which corresponds to
local SPDFT job-folder numbers [0, 2, 3] (confirmed via RD_type_info.txt,
which lists all 5 candidate types in folder-creation order: 0, 4, 11, 13, 15
-- Hindered types 4 and 15 dropped in the final Rotatable-only list).
"""
import os
import csv
import sys

sys.path.insert(0, os.path.expanduser("~/orca_runs"))
from parse_orca_geom_forces import ORCA_TO_GEMCIT_1BASED  # same permutation as the AIMD/Hessian pathway -- see that module for how it was derived/validated

HARTREE_TO_EV = 27.211386245988


def _remap_coords_to_gemcit_order(coords):
    """coords: list of (x,y,z) tuples in ORCA .inp atom order -> same list
    reordered into GEMCIT atom order."""
    out = [None] * len(coords)
    for orca_idx_1based, gemcit_idx_1based in ORCA_TO_GEMCIT_1BASED.items():
        out[gemcit_idx_1based - 1] = coords[orca_idx_1based - 1]
    return out

EQ_JOB_DIR = os.path.expanduser("~/orca_runs/hessian_disp_jobs/eq")
SPDFT_PARSED_DIR = os.path.expanduser("~/orca_runs/spdft_parsed")
OUT_DIR = os.path.expanduser("~/savesteps_run/gemcitabine/preprocessing_files")
MATERIAL = "GEMCIT"
LOCAL_FOLDER_ORDER = ["0", "2", "3"]  # matches global RD type order [0, 11, 13]


def parse_energy_hartree(job_out_path):
    energy = None
    with open(job_out_path) as f:
        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                energy = float(line.split()[-1])
    return energy


def parse_coords_from_inp(job_inp_path):
    with open(job_inp_path) as f:
        lines = f.readlines()
    coords = []
    in_block = False
    for line in lines:
        if line.strip().startswith("* xyz"):
            in_block = True
            continue
        if in_block:
            if line.strip() == "*":
                break
            parts = line.split()
            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return _remap_coords_to_gemcit_order(coords)


def main():
    geoms_out_path = os.path.join(OUT_DIR, f"{MATERIAL}_geoms_SPDFT.csv")
    e_out_path = os.path.join(OUT_DIR, f"{MATERIAL}_E_SPDFT.csv")

    n_geoms = 0
    with open(geoms_out_path, "w", newline="") as gf, open(e_out_path, "w", newline="") as ef:
        g_writer = csv.writer(gf)
        e_writer = csv.writer(ef)

        eq_energy_ha = parse_energy_hartree(os.path.join(EQ_JOB_DIR, "job.out"))
        eq_coords = parse_coords_from_inp(os.path.join(EQ_JOB_DIR, "job.inp"))
        e_writer.writerow([eq_energy_ha * HARTREE_TO_EV])
        for x, y, z in eq_coords:
            g_writer.writerow([x, y, z])
        n_geoms += 1
        print(f"equilibrium: E={eq_energy_ha * HARTREE_TO_EV:.6f} eV, {len(eq_coords)} atoms")

        for local_folder in LOCAL_FOLDER_ORDER:
            e_src = os.path.join(SPDFT_PARSED_DIR, f"{MATERIAL}_RDtype{local_folder}_E_SPDFT.csv")
            g_src = os.path.join(SPDFT_PARSED_DIR, f"{MATERIAL}_RDtype{local_folder}_geoms_SPDFT.csv")

            with open(e_src) as f:
                e_lines = f.readlines()
            with open(g_src) as f:
                g_lines = f.readlines()

            for line in e_lines:
                ef.write(line if line.endswith("\n") else line + "\n")
            for line in g_lines:
                gf.write(line if line.endswith("\n") else line + "\n")

            n_geoms += len(e_lines)
            print(f"local folder {local_folder}: {len(e_lines)} geometries appended")

    print(f"\nWrote {n_geoms} total geometries (1 equilibrium + {n_geoms - 1} displaced) to:")
    print(f"  {geoms_out_path}")
    print(f"  {e_out_path}")


if __name__ == "__main__":
    main()
