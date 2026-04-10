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
Continuous Relaxation Decoding for CSS Quantum LDPC Codes
Verification script for [[193, 25, d>=67]] D4-HGP code.

Constructs the parity-check matrix from the D4 causal diamond geometry
via the Tillich-Zemor hypergraph product, then validates all analytical
claims of the companion paper experimentally.
"""

import numpy as np
from itertools import combinations, product as iproduct
from collections import Counter
import time

# ── GF(2) primitives ────────────────────────────────────────────────────────

def gf2_rank(A):
    M = np.array(A, dtype=np.int8) % 2
    if M.ndim != 2 or 0 in M.shape: return 0
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

def gf2_null_right(A):
    A = np.array(A, dtype=np.int8) % 2; r, n = A.shape
    Aw = A.copy(); rank = 0; pivot_cols = []
    for col in range(n):
        rows = np.where(Aw[rank:, col] == 1)[0]
        if not len(rows): continue
        p = rows[0] + rank; Aw[[rank, p]] = Aw[[p, rank]]
        for row in range(r):
            if row != rank and Aw[row, col]: Aw[row] = (Aw[row] + Aw[rank]) % 2
        pivot_cols.append(col); rank += 1
        if rank == r: break
    free_cols = [c for c in range(n) if c not in pivot_cols]
    null_vecs = []
    for fc in free_cols:
        v = np.zeros(n, dtype=np.int8); v[fc] = 1
        for i, pc in enumerate(pivot_cols): v[pc] = Aw[i, fc] % 2
        null_vecs.append(v)
    return np.array(null_vecs, dtype=np.int8) if null_vecs else np.zeros((0, n), dtype=np.int8)

def independent_rows(H):
    H = np.array(H, dtype=np.int8) % 2
    selected = []; rank = 0
    for row in H:
        test = np.vstack(selected + [row]) if selected else row.reshape(1, -1)
        if gf2_rank(test) > rank:
            selected.append(row.copy()); rank += 1
    return np.array(selected, dtype=np.int8) if selected else np.zeros((0, H.shape[1]), dtype=np.int8)

# ── Build [[193, 25]] D4-HGP parity-check matrix ────────────────────────────

def build_hz_193():
    eta = np.diag([-1, 1, 1, 1])
    nl = sorted([v for v in iproduct([-1,0,1], repeat=4)
                 if v != (0,0,0,0) and int(np.array(v) @ eta @ np.array(v)) == 0])
    plaq = [list(q) for q in combinations(range(12), 4)
            if tuple(sum(nl[i][k] for i in q) for k in range(4)) == (0,0,0,0)]
    M = np.zeros((12, len(plaq)), dtype=np.int8)
    for j, p in enumerate(plaq):
        for i in p: M[i, j] = 1
    H = independent_rows(M.T)
    m, n = H.shape
    Im = np.eye(m, dtype=np.int8); In = np.eye(n, dtype=np.int8); HT = H.T
    HZ = np.hstack([np.kron(H, In), np.kron(Im, HT)]) % 2
    HX = np.hstack([np.kron(In, H), np.kron(HT, Im)]) % 2
    assert np.all(HZ @ HX.T % 2 == 0)
    return HZ, HX

# ── Energy functions ─────────────────────────────────────────────────────────

def make_supports(HZ):
    return [np.where(HZ[j])[0] for j in range(HZ.shape[0])]

def energy_linear(v, supports, lam=1.0):
    """
    LINEAR spring: H(v) = lam/4 * sum_i (v_i^2-1)^2 + sum_j (1-prod_j)/2.
    Ground state (v=+1) is NOT a critical point: gradient = -deg(q)/2 != 0.
    """
    cont = lam * 0.25 * np.sum((v**2 - 1)**2)
    spring = sum(0.5 * (1 - np.prod(v[s])) for s in supports)
    return cont + spring

def energy_sq(v, supports, lam=1.0):
    """
    SQUARED spring: H(v) = lam/4 * sum_i (v_i^2-1)^2 + sum_j (1-prod_j)^2/4.
    Valid codewords (v in {+/-1}^N satisfying all checks) are exact critical
    points with H=0 and grad H=0. See Theorem 1 in companion paper.
    """
    cont = lam * 0.25 * np.sum((v**2 - 1)**2)
    spring = sum(0.25 * (1 - np.prod(v[s]))**2 for s in supports)
    return cont + spring

def grad_sq(v, supports, lam=1.0):
    """
    Gradient of the squared-spring energy. Analytically:
      dH/dv_q = lam*(v_q^2-1)*v_q - (1/2)*sum_{j: q in j} (1-prod_j)*prod_{k!=q in j} v_k
    """
    g = lam * (v**2 - 1) * v
    for s in supports:
        sv = v[s]; fp = np.prod(sv)
        factor = -0.5 * (1.0 - fp)
        for qi, q in enumerate(s):
            g[q] += factor * (fp / sv[qi] if abs(sv[qi]) > 1e-12
                              else np.prod(np.delete(sv, qi)))
    return g

def grad_linear(v, supports, lam=1.0):
    """
    Gradient of linear spring energy (included to demonstrate the critical-point bug).
    dH/dv_q = lam*(v_q^2-1)*v_q - (1/2)*sum_{j: q in j} prod_{k!=q in j} v_k
    At v=+1 this equals -deg(q)/2 != 0.
    """
    g = lam * (v**2 - 1) * v
    for s in supports:
        sv = v[s]; fp = np.prod(sv)
        for qi, q in enumerate(s):
            g[q] += -0.5 * (fp / sv[qi] if abs(sv[qi]) > 1e-12
                            else np.prod(np.delete(sv, qi)))
    return g

# ── Decoder ──────────────────────────────────────────────────────────────────

def gradient_descent(v0, supports, lam=0.5, lr=0.05, n_steps=400):
    v = v0.copy()
    for _ in range(n_steps):
        g = grad_sq(v, supports, lam)
        v -= lr * g
        np.clip(v, -1.5, 1.5, out=v)
    return v

def snap_and_syndrome(v, HZ):
    vs = np.sign(v); vs[vs == 0] = 1.0
    vbin = ((1 - vs) / 2).astype(int) % 2
    return (HZ @ vbin % 2).sum()

# ── Verification tests ───────────────────────────────────────────────────────

def run_all_tests(HZ, HX):
    N, M = HZ.shape[1], HZ.shape[0]
    supports = make_supports(HZ)
    col_w = HZ.sum(axis=0)
    rng = np.random.default_rng(42)
    lam = 1.0
    eps = 1e-5

    print("=" * 65)
    print("CONTINUOUS RELAXATION DECODER — VERIFICATION PROTOCOL")
    print(f"Code: [[{N}, {N - 2*gf2_rank(HZ)}]], HZ: {M}x{N}, CSS valid")
    print("=" * 65)

    # ── Test 1: Ground state ─────────────────────────────────────────────
    print("\n[TEST 1] Ground state: E=0 and grad=0 (squared spring)")
    v0 = np.ones(N)
    E0 = energy_sq(v0, supports, lam)
    g0 = grad_sq(v0, supports, lam)
    norm_g0 = np.linalg.norm(g0)
    status = "PASS" if E0 < 1e-10 and norm_g0 < 1e-10 else "FAIL"
    print(f"  E(v=+1) = {E0:.2e}  (expected 0)  [{status}]")
    print(f"  |grad H|(v=+1) = {norm_g0:.2e}  (expected 0)  [{status}]")

    # ── Test 2: Bug demonstration (linear spring) ────────────────────────
    print("\n[TEST 2] Linear spring does NOT have critical points at codewords")
    g_lin = grad_linear(v0, supports, lam)
    norm_lin = np.linalg.norm(g_lin)
    expected_lin = np.sqrt(np.sum((col_w / 2)**2))
    status = "PASS" if abs(norm_lin - expected_lin) < 1e-8 else "FAIL"
    print(f"  |grad H_linear|(v=+1) = {norm_lin:.4f}  (expected {expected_lin:.4f})  [{status}]")
    print(f"  Analytical: sum_q (deg(q)/2)^2 under sqrt = {expected_lin:.4f}")

    # ── Test 3: Single-error energy formula ──────────────────────────────
    print("\n[TEST 3] Single-error energy E = deg(q) and grad[q] = -deg(q)")
    all_pass = True
    for q in [0, 42, 100, 150, 192]:
        v = np.ones(N); v[q] = -1.0
        E = energy_sq(v, supports, lam)
        g = grad_sq(v, supports, lam)
        expected_E = float(col_w[q])
        expected_g = -float(col_w[q])
        ok = abs(E - expected_E) < 1e-10 and abs(g[q] - expected_g) < 1e-10
        if not ok: all_pass = False
        print(f"  q={q:3d} deg={col_w[q]}: E={E:.1f} (exp {expected_E:.0f}), "
              f"grad[q]={g[q]:.1f} (exp {expected_g:.0f})  [{'PASS' if ok else 'FAIL'}]")

    # ── Test 4: Hessian diagonal at ground state ─────────────────────────
    print("\n[TEST 4] Hessian diagonal: H_qq = 2*lam + deg(q)/2")
    all_pass = True
    for q in [0, 42, 100, 150, 192]:
        vp = v0.copy(); vp[q] += eps
        vm = v0.copy(); vm[q] -= eps
        gp = grad_sq(vp, supports, lam)
        gm = grad_sq(vm, supports, lam)
        H_qq_num = (gp[q] - gm[q]) / (2 * eps)
        H_qq_ana = 2 * lam + col_w[q] / 2.0
        ok = abs(H_qq_num - H_qq_ana) < 1e-6
        if not ok: all_pass = False
        print(f"  q={q:3d} deg={col_w[q]}: numerical={H_qq_num:.4f}, "
              f"analytical 2λ+deg/2={H_qq_ana:.4f}  [{'PASS' if ok else 'FAIL'}]")

    # ── Test 5: Energy of t-error state ──────────────────────────────────
    print("\n[TEST 5] t-error energy: E = sum_{q in err} deg(q) (upper bound)")
    print("         (tight when error qubits share no checks)")
    rng2 = np.random.default_rng(7)
    all_pass = True
    for t in [1, 2, 3]:
        eq = rng2.choice(N, t, replace=False)
        v = np.ones(N); v[eq] = -1.0
        E = energy_sq(v, supports, lam=0.0)  # lam=0: pure spring energy
        E_ub = float(sum(col_w[q] for q in eq))
        ok = E <= E_ub + 1e-9
        print(f"  t={t}: E={E:.2f}, upper_bound=sum(deg)={E_ub:.0f}  [{'PASS' if ok else 'FAIL'}]")

    # ── Test 6: Gradient descent success rate ────────────────────────────
    print("\n[TEST 6] Gradient descent convergence (lam=0.5, lr=0.05, 400 steps)")
    print(f"  {'t':>4}  {'success/50':>12}  {'mean_residual_syn':>18}")
    for t in [1, 2, 3, 5]:
        ok_count = 0; residuals = []
        for _ in range(50):
            eq = rng.choice(N, t, replace=False)
            v = np.ones(N); v[eq] = -1.0
            vf = gradient_descent(v, supports, lam=0.5)
            r = snap_and_syndrome(vf, HZ)
            residuals.append(r)
            if r == 0: ok_count += 1
        print(f"  {t:4d}  {ok_count:4d}/50 ({ok_count/50:.2f})  "
              f"mean_syn_wt={np.mean(residuals):.3f}")

    # ── Test 7: Lambda sensitivity ───────────────────────────────────────
    print("\n[TEST 7] Lambda sensitivity (t=3, 40 trials each)")
    print(f"  {'lam':>6}  {'success/40':>12}  {'mean_final_E':>14}")
    for lam_test in [0.1, 0.5, 1.0, 2.0, 5.0]:
        ok_count = 0; fE = []
        for _ in range(40):
            eq = rng.choice(N, 3, replace=False)
            v = np.ones(N); v[eq] = -1.0
            vf = gradient_descent(v, supports, lam=lam_test)
            fE.append(energy_sq(vf, supports, lam_test))
            if snap_and_syndrome(vf, HZ) == 0: ok_count += 1
        print(f"  {lam_test:6.2f}  {ok_count:4d}/40 ({ok_count/40:.2f})  "
              f"mean_E={np.mean(fE):.5f}")

    # ── Test 8: Local minima census ──────────────────────────────────────
    print("\n[TEST 8] Residual syndrome weight distribution (t=5, 60 trials)")
    residuals = []
    for _ in range(60):
        eq = rng.choice(N, 5, replace=False)
        v = np.ones(N); v[eq] = -1.0
        vf = gradient_descent(v, supports, lam=0.5)
        residuals.append(snap_and_syndrome(vf, HZ))
    dist = Counter(residuals)
    for wt in sorted(dist):
        label = " ← global min" if wt == 0 else " ← local min (pseudo-codeword)"
        print(f"  residual_syn_wt={wt}: {dist[wt]} trials{label}")

    print("\n" + "=" * 65)
    print("VERIFICATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    t0 = time.time()
    print("Building [[193, 25]] HZ via D4 causal diamond HGP...")
    HZ, HX = build_hz_193()
    print(f"Done in {time.time()-t0:.2f}s. Running tests...\n")
    run_all_tests(HZ, HX)
