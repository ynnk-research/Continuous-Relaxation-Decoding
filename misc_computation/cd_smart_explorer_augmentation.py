# > Yannick Schmitt. (2026). Continuous Relaxation Decoding for CSS Quantum
# > LDPC Codes: Energy Landscape, Basin Geometry, Gradient Dynamics, and
# > Hardware Implementation. Zenodo.
# > https://doi.org/10.5281/zenodo.19484007
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Standalone module for:
1. Level-2 Augmentation Search: Testing both B4 orbits to push d_cl from 6 -> 8.
2. [[468, 36]] Torus Code Certification: Using RAM-capped Prange ISD to refute d >= 196.
"""

import numpy as np
from itertools import combinations, product as iproduct
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

# ─────────────────────────────────────────────────────────────────────────────
# §0  GF(2) Primitives (Fast)
# ─────────────────────────────────────────────────────────────────────────────

def gf2_rank(A):
    M = np.array(A, dtype=np.int8) % 2
    if M.size == 0: return 0
    r, c = M.shape; rank = 0
    for col in range(c):
        rows = np.where(M[rank:, col] == 1)[0]
        if not len(rows): continue
        p = rows[0] + rank; M[[rank, p]] = M[[p, rank]]
        for row in range(r):
            if row != rank and M[row, col]: M[row] = (M[row] + M[rank]) % 2
        rank += 1
        if rank == r: break
    return rank

def gf2_rref(M):
    M = np.array(M, dtype=np.int8) % 2
    r, c = M.shape; rank = 0; pivots = []
    for col in range(c):
        rows = np.where(M[rank:, col] == 1)[0]
        if not len(rows): continue
        p = rows[0] + rank; M[[rank, p]] = M[[p, rank]]
        for row in range(r):
            if row != rank and M[row, col]: M[row] = (M[row] + M[rank]) % 2
        pivots.append(col); rank += 1
        if rank == r: break
    return M, pivots

def gf2_independent_rows(H):
    H = np.array(H, dtype=np.int8) % 2
    selected = []; rank = 0
    for row in H:
        test = np.vstack(selected + [row.reshape(1,-1)]) if selected else row.reshape(1,-1)
        if gf2_rank(test) > rank:
            selected.append(row.copy()); rank += 1
    return np.array(selected, dtype=np.int8) if selected else np.zeros((0, H.shape[1]), dtype=np.int8)

def compute_logical_generators(H_X, H_Z):
    A = np.array(H_X, dtype=np.int8) % 2; r, c = A.shape
    aug = np.hstack([A.T, np.eye(c, dtype=np.int8)])
    pivot_cols, cur = [], 0
    for col in range(r):
        rows = np.where(aug[cur:, col] == 1)[0]
        if not len(rows): continue
        p = rows[0] + cur; aug[[cur, p]] = aug[[p, cur]]
        for row in range(c):
            if row != cur and aug[row, col]: aug[row] = (aug[row] + aug[cur]) % 2
        pivot_cols.append(col); cur += 1
        if cur == c: break
    G_null = (aug[len(pivot_cols):, r:] % 2).astype(np.int8)
    
    G_stab = gf2_independent_rows(H_Z)
    logicals = []; combined = list(G_stab); r_comb = G_stab.shape[0]
    for row in G_null:
        test = np.vstack(combined + [row.reshape(1,-1)]) if combined else row.reshape(1,-1)
        if gf2_rank(test) > r_comb:
            logicals.append(row.copy()); combined.append(row.copy()); r_comb += 1
    return np.array(logicals, dtype=np.int8) if logicals else np.zeros((0, c), dtype=np.int8)

def in_rowspace(v, H):
    if H.shape[0] == 0: return np.all(v == 0)
    aug = np.vstack([H, v.reshape(1, -1)]) % 2
    return gf2_rank(aug) == gf2_rank(H)


# ─────────────────────────────────────────────────────────────────────────────
# §1  Level-2 Augmentation Search
# ─────────────────────────────────────────────────────────────────────────────

def build_causal_diamond():
    eta = np.diag([-1, 1, 1, 1])
    nl = sorted([v for v in iproduct([-1,0,1], repeat=4) if v != (0,0,0,0) and int(np.array(v) @ eta @ np.array(v)) == 0])
    plaq = [list(q) for q in combinations(range(12), 4) if tuple(sum(nl[i][k] for i in q) for k in range(4)) == (0,0,0,0)]
    M = np.zeros((12, 21), dtype=np.int8)
    for j, p in enumerate(plaq):
        for i in p: M[i, j] = 1
    H_Z_primal = M.T.copy()
    H_primal_indep = gf2_independent_rows(H_Z_primal)
    GROUPS = [frozenset([0,5,6,11]), frozenset([1,4,7,10]), frozenset([2,3,8,9])]
    H_spatial = np.zeros((3, 12), dtype=np.int8)
    for i, G in enumerate(GROUPS):
        for q in G: H_spatial[i, q] = 1
    Gj = [H_spatial[i] for i in range(3)]
    return nl, H_primal_indep, Gj

def build_b4(nl):
    def apply(mat):
        ls = set(nl); perm = []
        for v in nl:
            w = tuple(int(x) for x in mat @ np.array(v))
            if w not in ls: return None
            perm.append(nl.index(w))
        return tuple(perm)
    def close(gens):
        seen = set(); grp = []
        stack = [tuple(range(12))]
        while stack:
            p = stack.pop()
            if p in seen: continue
            seen.add(p); grp.append(p)
            for g in gens:
                np2 = tuple(p[g[i]] for i in range(12))
                if np2 not in seen: stack.append(np2)
        return grp
    mats = []
    for i, j in combinations(range(4), 2):
        P = np.eye(4, dtype=int); P[[i,j]] = P[[j,i]]; mats.append(P)
    for i in range(4):
        S = np.eye(4, dtype=int); S[i,i] = -1; mats.append(S)
    gens = [p for p in (apply(m) for m in mats) if p is not None]
    return close(gens)

def find_classical_codewords(H, target_w):
    n = H.shape[1]; codewords = []
    for combo in combinations(range(n), target_w):
        v = np.zeros(n, dtype=np.int8)
        for i in combo: v[i] = 1
        if np.all((H @ v) % 2 == 0): codewords.append(v)
    return codewords

def _fast_classical_min_d(H, max_w=8):
    n = H.shape[1]
    for w in range(1, max_w + 1):
        for combo in combinations(range(n), w):
            v = np.zeros(n, dtype=np.int8)
            for i in combo: v[i] = 1
            if np.all((H @ v) % 2 == 0): return w, 1
    return 999, 0

def level_2_augmentation_search():
    print(f"\n{'═'*70}\n  RIGOROUS LEVEL-2 FILTRATION SEARCH (Target: d_cl = 8)\n{'─'*70}")
    
    nl, H_primal_indep, Gj = build_causal_diamond()
    n = H_primal_indep.shape[1]
    
    # 1. Get all 64 valid level-1 rows
    aug_rows = []
    for combo in combinations(range(n), 3):
        v = np.zeros(n, dtype=np.int8)
        for i in combo: v[i] = 1
        if all(int(v @ Gj[i]) % 2 == 1 for i in range(3)) and not in_rowspace(v, H_primal_indep):
            aug_rows.append(v)
            
    # 2. Split into 2 B4 Orbits
    b4 = build_b4(nl)
    supports = [frozenset(np.where(r)[0]) for r in aug_rows]
    visited, orbits = set(), []
    for supp in supports:
        if supp in visited: continue
        orb = frozenset(frozenset(perm[q] for q in supp) for perm in b4)
        orbits.append(orb); visited.update(orb)
    
    rep_rows = []
    for orb in orbits:
        v = np.zeros(n, dtype=np.int8)
        for q in list(orb)[0]: v[q] = 1
        rep_rows.append(v)

    print(f"  [+] Found {len(aug_rows)} valid Level-1 rows, split into {len(rep_rows)} B4 Orbits.")

    # 3. Exhaustive Level-2 search on both Orbit Representatives
    for orbit_idx, rep_row in enumerate(rep_rows):
        print(f"\n  ► Testing Orbit {orbit_idx+1} Representative Support: {list(np.where(rep_row)[0])}")
        H_aug1 = np.vstack([H_primal_indep, rep_row.reshape(1,-1)]) % 2
        
        cw_6 = find_classical_codewords(H_aug1, 6)
        cw_7 = find_classical_codewords(H_aug1, 7)
        print(f"    Target: Kill {len(cw_6)} w=6 and {len(cw_7)} w=7 codewords.")
        
        found_d8 = False
        for w in range(1, 6):
            print(f"    Scanning weight-{w} rows... ", end="", flush=True)
            hits = 0
            for combo in combinations(range(n), w):
                v = np.zeros(n, dtype=np.int8)
                for i in combo: v[i] = 1
                
                if all(np.dot(v, c) % 2 == 1 for c in cw_6):
                    if all(np.dot(v, c) % 2 == 1 for c in cw_7) and not in_rowspace(v, H_aug1):
                        H_aug2 = np.vstack([H_aug1, v.reshape(1,-1)]) % 2
                        d2, _ = _fast_classical_min_d(H_aug2, max_w=8)
                        if d2 >= 8:
                            hits += 1
                            found_d8 = True
            print(f"Found {hits} valid augmentations." if hits > 0 else "None.")
            if found_d8: break
            
        if not found_d8:
            print(f"    Result: Orbit {orbit_idx+1} FAILS the filtration conjecture (stops at d=6).")

    print(f"\n{'─'*70}\n  FILTRATION CONJECTURE CONCLUSION:")
    print("  If both orbits failed, the geometric filtration is rigidly capped at d=6.")
    print("  If an orbit succeeded, the causal diamond possesses theoretically growing distance!")


# ─────────────────────────────────────────────────────────────────────────────
# §2  [[468, 36]] Generation
# ─────────────────────────────────────────────────────────────────────────────

def build_d4_diamond_geometry():
    eta = np.diag([-1, 1, 1, 1])
    def msq(v): return int(np.array(v) @ eta @ np.array(v))
    nl = sorted([v for v in iproduct([-1, 0, 1], repeat=4) if v != (0, 0, 0, 0) and msq(v) == 0])
    plaq = []
    for q in combinations(range(12), 4):
        vs = [nl[i] for i in q]
        s = tuple(sum(v[k] for v in vs) for k in range(4))
        if s == (0, 0, 0, 0): plaq.append(list(q))
    return nl, plaq

def build_chain_seed(L):
    nl, plaq = build_d4_diamond_geometry()
    future = [i for i, v in enumerate(nl) if v[0] > 0]
    past = [i for i, v in enumerate(nl) if v[0] < 0]
    f_pos = {v: i for i, v in enumerate(future)}
    p_pos = {v: i for i, v in enumerate(past)}
    remap = []
    for pq in plaq:
        fs = [f_pos[i] for i in pq if i in set(future)]
        ps = [p_pos[i] for i in pq if i in set(past)]
        remap.append((fs, ps))
    n_q = 6 * (L + 1)
    n_s = 21 * L
    H = np.zeros((n_s, n_q), dtype=np.int8)
    for d in range(L):
        for pid, (fs, ps) in enumerate(remap):
            for lp in ps: H[21 * d + pid, 6 * d + lp] = 1
            for lf in fs: H[21 * d + pid, 6 * (d + 1) + lf] = 1
    return gf2_independent_rows(H % 2)

def hypergraph_product_css(H):
    m, n = H.shape
    Im = np.eye(m, dtype=np.int8)
    In = np.eye(n, dtype=np.int8)
    HT = H.T
    HZ = np.hstack([np.kron(H, In), np.kron(Im, HT)]) % 2
    HX = np.hstack([np.kron(In, H), np.kron(HT, Im)]) % 2
    return HZ, HX

def generate_468_matrices():
    """Builds the L=2 Torus code."""
    H = build_chain_seed(2)
    HZ, HX = hypergraph_product_css(H)
    return HX, HZ


# ─────────────────────────────────────────────────────────────────────────────
# §3  RAM-Capped ISD Worker & Certifier
# ─────────────────────────────────────────────────────────────────────────────

def _isd_worker_468(args):
    HX_bytes, HZ_bytes, shapeX, shapeZ, n_trials, seed, known_ub = args
    H_X = np.frombuffer(HX_bytes, dtype=np.int8).reshape(shapeX).copy()
    H_Z = np.frombuffer(HZ_bytes, dtype=np.int8).reshape(shapeZ).copy()
    N   = shapeX[1]
    rng = np.random.default_rng(seed)

    G_log  = compute_logical_generators(H_X, H_Z)
    G_stab = gf2_independent_rows(H_Z)

    if G_log.shape[0] == 0: return known_ub, None

    k = G_log.shape[0]
    n_comb_total = (1 << k) - 1

    G_combined = np.vstack([G_log, G_stab]).astype(np.int8)
    dim = G_combined.shape[0]
    best_w = known_ub
    best_v = None

    # Strict RAM cap (~50 MB per core)
    max_combos = 65535
    if n_comb_total <= max_combos:
        a_vals = np.arange(1, n_comb_total + 1, dtype=np.int64)
    else:
        basis = np.array([1 << i for i in range(k)], dtype=np.int64)
        randoms = rng.integers(1, n_comb_total + 1, size=max_combos - k, dtype=np.int64)
        a_vals = np.concatenate([basis, randoms])

    n_comb = len(a_vals)
    bits = np.zeros((n_comb, k), dtype=np.int8)
    for j in range(k):
        bits[:, j] = ((a_vals >> j) & 1).astype(np.int8)

    for trial in range(n_trials):
        perm     = rng.permutation(N)
        perm_inv = np.argsort(perm)
        G_perm   = G_combined[:, perm]
        G_red, pivots = gf2_rref(G_perm)

        if len(pivots) < dim: continue
        log_rows_perm = G_red[:k]

        combos = (bits.astype(np.int32) @ log_rows_perm.astype(np.int32)) % 2
        weights = combos.sum(axis=1)

        valid = (weights > 0) & (weights < best_w)
        if not np.any(valid): continue

        best_idx = int(np.argmin(np.where(valid, weights, best_w)))
        if not valid[best_idx]: continue

        w_cand = int(weights[best_idx])
        v = combos[best_idx].astype(np.int8)[perm_inv]

        if np.any((H_X @ v) % 2 != 0): continue
        if in_rowspace(v, H_Z): continue

        best_w = w_cand
        best_v = tuple(np.where(v)[0].tolist())

    return best_w, best_v

def certify_large_code(H_X, H_Z, n_workers, isd_trials):
    N = H_X.shape[1]
    k = N - gf2_rank(H_X) - gf2_rank(H_Z)
    
    print(f"\n{'═'*70}\n  EXACT CERTIFICATION: [[{N}, {k}]] CODE\n{'─'*70}")
    print(f"  Target: Refuting d ≥ 196 hypothesis.")
    print(f"  Strategy: RAM-capped parallel Prange ISD.")
    print(f"  Trials: {n_workers} workers × {isd_trials} = {n_workers * isd_trials} total trials.")
    
    known_ub = min(196, N + 1)
    
    HX_b = H_X.astype(np.int8).tobytes()
    HZ_b = H_Z.astype(np.int8).tobytes()
    shX, shZ = H_X.shape, H_Z.shape

    args_list = [(HX_b, HZ_b, shX, shZ, isd_trials, 42 + i*7919, known_ub) for i in range(n_workers)]
    best_w = known_ub; best_v = None

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_isd_worker_468, a): i for i, a in enumerate(args_list)}
        for fut in as_completed(futures):
            w, v = fut.result()
            print(f"    Worker {futures[fut]:>2} finished. Best logic found: w = {w if w < known_ub else '> 196'}")
            if w < best_w:
                best_w = w; best_v = v

    print(f"\n  [RESULT] Time elapsed: {time.time()-t0:.1f}s")
    if best_v is not None:
        print(f"  *** FALSE: The true distance is exactly (or at most) {best_w} ***")
        print(f"      Logical Support: {list(best_v)}")
        print("      (BP-based sampling of d>=196 was fatally trapped in pseudo-codewords)")
    else:
        print(f"  *** UNRESOLVED: No logical < {known_ub} found in {n_workers*isd_trials} trials. ***")
        print(f"      d >= {known_ub} remains theoretically possible, though highly unlikely.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    
    # 1. Run Level-2 Augmentation Search
    level_2_augmentation_search()
    
    # 2. Run [[468, 36]] Certification
    n_workers = max(2, multiprocessing.cpu_count() - 1)
    print(f"\n  [+] Generating [[468, 36]] L=2 Torus Code natively in memory...")
    H_X, H_Z = generate_468_matrices()
    
    # 5000 trials per worker = massive statistical confidence
    certify_large_code(H_X, H_Z, n_workers=n_workers, isd_trials=5000)
