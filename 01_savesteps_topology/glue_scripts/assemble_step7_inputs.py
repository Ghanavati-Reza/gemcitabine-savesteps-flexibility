import os
import shutil

HOME = os.path.expanduser("~")
TYPING_SRC = os.path.join(HOME, "savesteps_run/gemcitabine/preprocessing_files/Materials_Hindered_rotatable_inputs/GEMCIT")
GEOM_FORCES_SRC = os.path.join(HOME, "orca_runs/geom_forces_parsed")
PREPROC = os.path.join(HOME, "savesteps_run/gemcitabine/preprocessing_files")

OUTDIR = os.path.join(HOME, "savesteps_run/gemcitabine/Flexibility_parameters_calculations_final_input_files/GEMCIT/Average/geometry_and_typing_files")
os.makedirs(OUTDIR, exist_ok=True)

files_to_copy = [
    (os.path.join(TYPING_SRC, "GEMCIT_angle_instances_list.txt"), "GEMCIT_angle_instances_list.txt"),
    (os.path.join(TYPING_SRC, "GEMCIT_angle_types_list.txt"), "GEMCIT_angle_types_list.txt"),
    (os.path.join(TYPING_SRC, "GEMCIT_atom_types_with_coordinates.txt"), "GEMCIT_atom_types_with_coordinates.txt"),
    (os.path.join(TYPING_SRC, "GEMCIT_bond_and_UB_instances_list.txt"), "GEMCIT_bond_and_UB_instances_list.txt"),
    (os.path.join(TYPING_SRC, "GEMCIT_bond_and_UB_types_list.txt"), "GEMCIT_bond_and_UB_types_list.txt"),
    (os.path.join(TYPING_SRC, "GEMCIT_dihedral_instances_list.txt"), "GEMCIT_dihedral_instances_list.txt"),
    (os.path.join(TYPING_SRC, "GEMCIT_dihedral_types_list.txt"), "GEMCIT_dihedral_types_list.txt"),
    (os.path.join(TYPING_SRC, "GEMCIT.xyz"), "GEMCIT.xyz"),
    (os.path.join(GEOM_FORCES_SRC, "geom_forces_array_GEMCIT_w_Hess.csv"), "geom_forces_array_GEMCIT_w_Hess.csv"),
    (os.path.join(GEOM_FORCES_SRC, "geom_forces_array_GEMCIT_val.csv"), "geom_forces_array_GEMCIT_val.csv"),
    (os.path.join(PREPROC, "GEMCIT_geoms_SPDFT.csv"), "GEMCIT_geoms_SPDFT.csv"),
    (os.path.join(PREPROC, "GEMCIT_E_SPDFT.csv"), "GEMCIT_E_SPDFT.csv"),
    (os.path.join(PREPROC, "RD_type_info_final.txt"), "RD_type_ind_SPDFT.txt"),
]

for src, dst_name in files_to_copy:
    dst = os.path.join(OUTDIR, dst_name)
    if not os.path.exists(src):
        print(f"MISSING SOURCE: {src}")
        continue
    shutil.copy(src, dst)
    print(f"copied -> {dst_name}")

print("\nFinal directory contents:")
for f in sorted(os.listdir(OUTDIR)):
    print(" ", f)
