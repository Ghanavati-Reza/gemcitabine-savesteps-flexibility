import os
import shutil

SRC = os.path.expanduser("~/savesteps_run/gemcitabine/preprocessing_files/Materials_Hindered_rotatable_inputs_with_counts/GEMCIT")
DST = os.path.expanduser("~/savesteps_run/gemcitabine/Flexibility_parameters_calculations_final_input_files/GEMCIT/Average/geometry_and_typing_files")

files = [
    "GEMCIT_angle_instances_list.txt",
    "GEMCIT_angle_types_list.txt",
    "GEMCIT_atom_types_with_coordinates.txt",
    "GEMCIT_bond_and_UB_instances_list.txt",
    "GEMCIT_bond_and_UB_types_list.txt",
    "GEMCIT_dihedral_instances_list.txt",
    "GEMCIT_dihedral_types_list.txt",
    "GEMCIT.xyz",
]

for f in files:
    src = os.path.join(SRC, f)
    dst = os.path.join(DST, f)
    shutil.copy(src, dst)
    print(f"copied {f}: {os.path.getsize(dst)} bytes")
