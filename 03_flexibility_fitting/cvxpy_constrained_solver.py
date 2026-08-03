"""
PSD-constrained LASSO solver for the SAVESTEPS bonded/cross-term fit.

Instead of glmnet's unconstrained-except-box-bounds LASSO followed by a
post-hoc "check the local coupling Hessians, drop the ones that aren't
positive-definite, then empirically sweep a margin against actual LAMMPS MD
runs" pipeline, this fits the SAME weighted L1-penalized least-squares
objective but with the coupling Hessians constrained to be positive-definite
(via a small fixed numerical margin EPS, NOT an empirically-tuned knob) as
part of the convex program itself.

HISTORY (why this is a single GLOBAL constraint, not per-angle or per-atom):
  1. First attempt: one 3x3/2x2 PSD block per angle/angle-pair. Necessary but
     not sufficient -- an atom with valence >= 4 has multiple angles sharing
     its bonds, and several individually-PD blocks can have an indefinite
     COMBINED effect on a shared bond. Confirmed via a real LAMMPS blowup.
  2. Second attempt: one joint PSD block per ATOM (all its bonds + angles +
     cross terms). Still not sufficient -- every bond is shared by exactly
     two atoms, so it appears (with its full stiffness) in both atoms'
     blocks independently; verifying each block separately does not
     reconcile how that shared stiffness is used across both, so the full
     assembled molecular Hessian still isn't guaranteed PSD. Also confirmed
     via a real LAMMPS blowup (different bond, same failure mode).
  3. This version: ONE global Hessian spanning every bond instance and every
     angle instance in the molecule simultaneously (~80x80 for a small
     molecule), with every bond_bond/bond_angle/angle_angle cross term
     placed as an off-diagonal entry. A single PSD constraint on this whole
     matrix is the only construction that's actually rigorous, since the
     molecule's bonded topology is one connected graph -- no smaller
     decomposition avoids overlapping shared coordinates.

Reformulated in "sufficient statistics" form (A = X^T W X, c = X^T W y) so
the conic solver (Clarabel/SCS) only ever sees a compact ~285-variable QP
with a single (~80x80) LMI constraint, never the full ~100k-row design
matrix.
"""
import re
import ast

import numpy as np
import cvxpy as cp

EPS = 1e-3  # fixed numerical margin for "> 0" eigenvalues -- not tuned against MD results


def _parse_bracket_line(line):
    line = line.strip()
    return ast.literal_eval(line) if line else None


def _parse_instances(path):
    out = []
    with open(path) as f:
        for line in f:
            v = _parse_bracket_line(line)
            if v is not None:
                out.append(v)
    return out


def _load_topology(base_dir):
    bond_instances = _parse_instances(f"{base_dir}/GEMCIT_bond_and_UB_instances_list.txt")
    angle_instances = _parse_instances(f"{base_dir}/GEMCIT_angle_instances_list.txt")

    # bond instance: [atomA, imgA, atomB, imgB, type_idx, ...]
    bonds = [(b[0], b[2], b[4]) for b in bond_instances]  # (atomA, atomB, type)
    # angle instance: [center, term1, img1, term2, img2, type_idx, ...]
    angles = [(a[1], a[0], a[3], a[5]) for a in angle_instances]  # (term1, center, term2, type)

    return bonds, angles


def _build_global_block(bonds, angles, stretch_col, bend_col, bbc_col, bac_col, aac_col):
    """Assemble the single molecule-wide Hessian descriptor: one row/column
    per PHYSICAL bond instance, then one per physical angle instance (the
    two are 1:1 with TYPES for GEMCIT's angles, and nearly 1:1 for bonds --
    a few bond types have 2 equivalent instances, each a separate DOF with
    the same k). Cross terms are looked up per angle INSTANCE (its specific
    two flanking bond instances), not per type, so this is correct even if
    a type has multiple geometrically-unrelated instances."""
    nb = len(bonds)
    na = len(angles)
    n = nb + na

    bond_instance_type = [t for (_, _, t) in bonds]
    bond_idx_by_pair = {frozenset([a, b]): i for i, (a, b, t) in enumerate(bonds)}

    stretch_idx = [stretch_col[t] for t in bond_instance_type]
    bend_idx = []
    bbc_idx = []
    bac_idx = []
    flank_bond_idx = []  # (bond_i, bond_j) global bond-row indices flanking this angle instance
    for (t1_atom, center, t2_atom, atype) in angles:
        bend_idx.append(bend_col[atype])
        bbc_idx.append(bbc_col.get(atype))
        bac_idx.append(bac_col.get(atype))
        bi = bond_idx_by_pair.get(frozenset([center, t1_atom]))
        bj = bond_idx_by_pair.get(frozenset([center, t2_atom]))
        flank_bond_idx.append((bi, bj))

    # angle-angle: couples two angle INSTANCES sharing a vertex atom, whose
    # TYPES have a fitted angle_angle column.
    aa_pairs = []  # (angle_row_i, angle_row_j, column_index)
    for i in range(na):
        for j in range(i + 1, na):
            _, ci, _, ti = angles[i]
            _, cj, _, tj = angles[j]
            if ci != cj:
                continue
            tp = (min(ti, tj), max(ti, tj)) if ti != tj else None
            col = aac_col.get(tp) if tp is not None else None
            if col is not None:
                aa_pairs.append((i, j, col))

    return {
        "nb": nb, "na": na, "n": n,
        "stretch_idx": stretch_idx, "bend_idx": bend_idx,
        "bbc_idx": bbc_idx, "bac_idx": bac_idx,
        "flank_bond_idx": flank_bond_idx, "aa_pairs": aa_pairs,
    }


def _assemble_matrix(blk, get):
    """Build the n x n symmetric matrix. `get(idx)` maps a column index (or
    None) to either a cvxpy scalar expression or a python float, used for
    both the cvxpy constraint and the numpy verification."""
    n = blk["n"]
    nb = blk["nb"]
    M = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(nb):
        M[i][i] = get(blk["stretch_idx"][i])
    for k in range(blk["na"]):
        # NOT 2*k: manz_bend's near-equilibrium form is U ~= (1/2)*k*(theta-theta0)^2
        # (verified numerically via finite-difference on the actual RASPA MANZ_BEND
        # formula, d^2U/dtheta^2 = k exactly at theta0), same "coefficient IS the
        # true curvature" convention as bonds and cross terms. A stray `2*` here
        # previously overstated every bend's diagonal entry.
        M[nb + k][nb + k] = get(blk["bend_idx"][k])
    for k in range(blk["na"]):
        bi, bj = blk["flank_bond_idx"][k]
        if bi is None or bj is None:
            continue
        val = get(blk["bbc_idx"][k])
        M[bi][bj] = val
        M[bj][bi] = val
        val = get(blk["bac_idx"][k])
        M[bi][nb + k] = val
        M[nb + k][bi] = val
        M[bj][nb + k] = val
        M[nb + k][bj] = val
    for (i, j, col) in blk["aa_pairs"]:
        val = get(col)
        M[nb + i][nb + j] = val
        M[nb + j][nb + i] = val
    return M


def _to_1x1(e):
    if isinstance(e, (int, float)):
        return np.zeros((1, 1)) if e == 0 else np.full((1, 1), float(e))
    return cp.reshape(e, (1, 1), order='F')


def _psd_constraint(bvar, blk):
    def get(idx):
        return bvar[idx] if idx is not None else 0
    M = _assemble_matrix(blk, get)
    n = blk["n"]
    H = cp.bmat([[_to_1x1(M[i][j]) for j in range(n)] for i in range(n)])
    return H >> EPS * np.eye(n)


def _min_eig(bv, blk):
    def get(idx):
        return bv[idx] if idx is not None else 0.0
    M = np.array(_assemble_matrix(blk, get), dtype=float)
    return np.linalg.eigvalsh(M).min()


def _verify_and_fix_psd(bv, blk):
    """Post-threshold numpy safety check on the single global block: if it's
    no longer PSD after thresholding, zero cross terms in order of smallest
    magnitude first until it is again. Zeroing every cross term always
    trivially restores PSD-ness (nonneg diagonal only, guaranteed by the box
    constraint), so this can never fail to terminate."""
    cross_idx = []
    for k in range(blk["na"]):
        if blk["bbc_idx"][k] is not None:
            cross_idx.append(blk["bbc_idx"][k])
        if blk["bac_idx"][k] is not None:
            cross_idx.append(blk["bac_idx"][k])
    for (_, _, col) in blk["aa_pairs"]:
        cross_idx.append(col)
    cross_idx = sorted(set(cross_idx), key=lambda idx: abs(bv[idx]))

    if _min_eig(bv, blk) >= -1e-9:
        return
    for idx in cross_idx:
        if bv[idx] == 0.0:
            continue
        bv[idx] = 0.0
        if _min_eig(bv, blk) >= -1e-9:
            return


def solve(export, base_dir, lambda_grid_points=40, solver="CLARABEL", verbose=False):
    X = export["X_array"]
    y = export["forces_energies"].reshape(-1)
    c1_final = export["c1_final"]
    PF = export["PF_final"].reshape(-1)
    w = export["weights"].reshape(-1)
    natoms = export["natoms"]

    nstrech = export["nstrech_potential"]
    nbend = export["nbend_potential"]
    nbb = export["nbond_bond_potential"]
    nba = export["nbond_angle_potential"]
    naa = export["n_angle_angle_potential"]

    ncols = X.shape[1]
    assert PF.shape[0] == ncols

    # ---- column index maps ----
    stretch_types = list(export["list_stretch_type_info_ks_0"])
    bend_types = list(export["list_bend_type_info_ks_0"])
    bbc_types = list(export["list_bbc_type_info_ks_0"])
    bac_types = list(export["list_bac_type_info_ks_0"])
    aac_names = list(export["list_aac_type_info_ks_0"])

    stretch_col = {t: i for i, t in enumerate(stretch_types)}
    bend_col = {t: nstrech + i for i, t in enumerate(bend_types)}
    bbc_col = {t: nstrech + nbend + i for i, t in enumerate(bbc_types)}
    bac_col = {t: nstrech + nbend + nbb + i for i, t in enumerate(bac_types)}

    aac_col = {}
    for i, name in enumerate(aac_names):
        m = re.match(r"angle_type_(\d+)_x_angle_type_(\d+)", name)
        t1, t2 = int(m.group(1)), int(m.group(2))
        aac_col[(t1, t2)] = nstrech + nbend + nbb + nba + i

    bonds, angles = _load_topology(base_dir)
    blk = _build_global_block(bonds, angles, stretch_col, bend_col, bbc_col, bac_col, aac_col)
    if verbose:
        print(f"global PSD block: {blk['nb']} bond instances + {blk['na']} angle instances = {blk['n']}x{blk['n']}")

    # ---- sufficient statistics (collapses the ~100k-row problem to ncols x ncols) ----
    Xw = X * w[:, None]
    A = X.T @ Xw          # ncols x ncols, PSD
    cvec = X.T @ (w * y)  # ncols,
    yty = float(np.sum(w * y * y))
    A = 0.5 * (A + A.T)   # symmetrize away floating-point asymmetry

    b = cp.Variable(ncols)

    # ---- box constraints (same as glmnet's cl argument) ----
    constraints = []
    lb = c1_final[0, :]
    finite_lb = np.isfinite(lb)
    if finite_lb.any():
        constraints.append(b[finite_lb] >= lb[finite_lb])

    # ---- single global PSD constraint (see module docstring) ----
    constraints.append(_psd_constraint(b, blk))

    # ---- lambda path (largest -> smallest, matching glmnet's lambdau convention) ----
    grad0 = cvec  # gradient of 0.5*b'Ab - c'b at b=0 is -c
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.abs(grad0) / np.where(PF > 0, PF, np.inf)
    lambda_max = float(np.max(ratios)) * 1.05
    lambda_min = lambda_max * 1e-4
    lambdas = np.geomspace(lambda_max, lambda_min, lambda_grid_points)

    results = []  # (lam, b_val, n_params, weighted_R2)
    prob_param = cp.Parameter(nonneg=True)
    # A = X^T W X is PSD by construction, but its size/conditioning makes
    # cvxpy's default iterative PSD certification (ARPACK) fail to converge;
    # psd_wrap skips that check since we already know it holds mathematically.
    objective = cp.Minimize(0.5 * cp.quad_form(b, cp.psd_wrap(A)) - cvec @ b
                             + prob_param * cp.sum(cp.multiply(PF, cp.abs(b))))
    problem = cp.Problem(objective, constraints)

    # Thresholding: cross-term (off-diagonal) columns are thresholded freely
    # against their own natural LASSO scale (lambda*PF_j). Diagonal
    # (stretch/bend) columns only get a tiny numerical-noise cleanup, since a
    # diagonal entry may be load-bearing for a surviving cross term. The
    # global eigenvalue check afterward is the actual guarantee.
    THRESH_FACTOR = 0.05
    DIAG_EPS = 1e-6

    diag_cols = set(stretch_col.values()) | set(bend_col.values())
    cross_cols = set(bbc_col.values()) | set(bac_col.values()) | set(aac_col.values())

    for lam in lambdas:
        prob_param.value = float(lam)
        try:
            problem.solve(solver=solver, verbose=False)
        except cp.error.SolverError:
            problem.solve(solver="SCS", verbose=False)
        if b.value is None:
            if verbose:
                print(f"lambda={lam:.6g}: solver FAILED, skipping")
            continue
        bv = np.array(b.value).reshape(-1)

        natural_scale = lam * PF
        for j in range(ncols):
            if j in diag_cols:
                if abs(bv[j]) < DIAG_EPS:
                    bv[j] = 0.0
            elif j in cross_cols:
                if abs(bv[j]) < THRESH_FACTOR * max(natural_scale[j], 1e-12):
                    bv[j] = 0.0

        _verify_and_fix_psd(bv, blk)

        n_params = int(np.count_nonzero(bv))
        rss = float(bv @ A @ bv - 2 * cvec @ bv + yty)
        r2 = 1.0 - rss / yty if yty > 0 else 0.0
        results.append((float(lam), bv, n_params, r2))
        if verbose:
            print(f"lambda={lam:.6g}: n_params={n_params}, R2={r2:.6f}")

    if not results:
        raise RuntimeError("no lambda in the path produced a feasible solve")

    # ---- lambda_final selection: identical backward-elimination criterion
    # to fit_cross.py's original glmnet-path walk (walk from the densest/
    # smallest-lambda end toward sparser models; stop at the first point
    # where dropping further params would cost too much R^2 per param). ----
    lam_j, b_j, n_params_j, r2_j = results[-1]
    lambda_final = lam_j
    b_final_dense = results[-1][1]  # "lambda -> 0" analogue, informational only
    b_selected = b_j
    for i in range(len(results) - 2, -1, -1):
        lam_i, b_i, n_params_i, r2_i = results[i]
        if n_params_i < n_params_j:
            denom_r2 = (1 - r2_i)
            if denom_r2 <= 0 or n_params_j == n_params_i:
                continue
            temp = (3 * natoms / denom_r2) * ((r2_j - r2_i) / (n_params_j - n_params_i))
            if temp > 0.5:
                lambda_final = lam_j
                b_selected = b_j
                break
            else:
                lam_j, b_j, n_params_j, r2_j = lam_i, b_i, n_params_i, r2_i
    else:
        b_selected = b_j
        lambda_final = lam_j

    if verbose:
        print(f"selected lambda_final={lambda_final:.6g}, n_params={int(np.count_nonzero(b_selected))}")
        print(f"verification: min eigenvalue of the global {blk['n']}x{blk['n']} block at b_final = {_min_eig(b_selected, blk):.6g}")

    b_final_0 = b_final_dense.reshape(-1, 1)
    b_final = b_selected.reshape(-1, 1)
    info = {
        "lambda_final": float(lambda_final),
        "n_params_final": int(np.count_nonzero(b_selected)),
        "lambda_at_dense_end": float(results[-1][0]),
        "n_params_at_dense_end": int(np.count_nonzero(b_final_dense)),
    }
    return b_final_0, b_final, info
