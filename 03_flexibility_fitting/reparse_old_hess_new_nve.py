import sys, os, re
sys.path.insert(0, os.path.expanduser('~/orca_runs'))
from parse_orca_geom_forces import collect_engrad_jobs, write_geom_forces_csv

material = 'GEMCIT'
out_dir = os.path.expanduser('~/orca_runs/geom_forces_parsed')
os.makedirs(out_dir, exist_ok=True)

hess_dir = os.path.expanduser('~/orca_runs/hessian_disp_jobs')  # OLD Hessian: delta=0.01 A, 2-point (+/-)
hess_paths = collect_engrad_jobs(hess_dir)
print(f'Found {len(hess_paths)} old-Hessian .engrad files (expect 175, incl eq)')

aimd_training_dir = os.path.expanduser('~/orca_runs/aimd_nve_training')
aimd_val_dir = os.path.expanduser('~/orca_runs/aimd_nve_validation')


def natural_mdstep_key(p):
    m = re.search(r'mdstep(\d+)', os.path.basename(p))
    return int(m.group(1)) if m else 0


def collect_aimd_paths(aimd_dir):
    paths = []
    for traj in sorted(os.listdir(aimd_dir)):
        traj_dir = os.path.join(aimd_dir, traj)
        if os.path.isdir(traj_dir):
            md_paths = sorted(
                (f for f in os.listdir(traj_dir) if f.startswith('mdstep') and f.endswith('.engrad')),
                key=natural_mdstep_key,
            )
            paths.extend(os.path.join(traj_dir, f) for f in md_paths)
    return paths


training_aimd_paths = collect_aimd_paths(aimd_training_dir)
print(f'Found {len(training_aimd_paths)} NVE AIMD training .engrad files (expect 1000)')

training_paths = hess_paths + training_aimd_paths
print(f'Total training geometries (old Hessian + NEW NVE AIMD): {len(training_paths)} (expect 175+1000=1175)')

write_geom_forces_csv(training_paths, os.path.join(out_dir, f'geom_forces_array_{material}_w_Hess_oldhess_newnve.csv'))

val_paths = collect_aimd_paths(aimd_val_dir)
print(f'Found {len(val_paths)} NVE AIMD validation .engrad files (expect 1000)')
write_geom_forces_csv(val_paths, os.path.join(out_dir, f'geom_forces_array_{material}_val_newnve.csv'))
