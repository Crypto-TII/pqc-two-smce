#!/usr/bin/env sage -python
"""SageMath implementation of Algorithm 1 (Figure 1): solver for the 2-sMCE problem.

Instance:  G1 = S1 G0 Q,  G3 = S2 G2 Q,  Q = A^T (x) B,  A^T = eps_A A,  B^T = eps_B B,
with unrelated changes of basis S1, S2 in GL_k(F_q).  An sIMCE instance (G0, G1, G2) with
G1 = S1 G0 Q and G2 = S2 G0 Q^{-1} is the 2-sMCE instance (G0, G1, G2, G0).

Usage:
    sage -python algorithm1.py [--family 2-sMCE|sIMCE] [--k 8 --m 8 --n 8] [--q 4093]
                               [--eps-a 1|-1 --eps-b 1|-1] [--seed 0]
"""

import argparse
import os
import random
import time

from sage.all import GF, block_matrix, identity_matrix, matrix, vector, zero_matrix

SPARSE_BELOW = int(os.environ.get("SMCE_SPARSE_BELOW", 1021))


def _sparse(F):
    return F.is_finite() and F.order() < SPARSE_BELOW


def mat(F, nrows, ncols, entries=None):
    """Matrix constructor honouring the sparse workaround."""
    if entries is None:
        return zero_matrix(F, nrows, ncols, sparse=_sparse(F))
    return matrix(F, nrows, ncols, entries, sparse=_sparse(F))


def kron(A, B):
    return A.tensor_product(B, subdivide=False)


def sym_dimension(t, eps):
    return t * (t + eps) // 2


def duplication(F, t, eps):
    """t^2 x d_eps(t) matrix D with vec(X) = D x for X in S_eps(t) and x its free entries (i <= j; i < j if skew)."""
    cols = []
    for i in range(t):
        for j in range(i if eps == 1 else i + 1, t):
            c = [0] * (t * t)
            c[i * t + j] = 1
            if i != j:
                c[j * t + i] = eps
            cols.append(c)
    return mat(F, len(cols), t * t, cols).transpose()


def reshape(row, m, n):
    """Row vector of length mn -> m x n matrix (row-major)."""
    return mat(row.base_ring(), m, n, list(row))


# ----------------------------------------------------------------------------- instances
def random_symmetric_invertible(F, t, eps, rng):
    """Uniformly random invertible matrix in S_eps(t) (skew-symmetric needs t even)."""
    if eps == -1 and t % 2 == 1:
        raise ValueError("no invertible skew-symmetric matrix of odd size")
    while True:
        M = mat(F, t, t, [F(rng.randrange(F.order())) for _ in range(t * t)])
        X = M + eps * M.transpose() if eps == 1 else M - M.transpose()
        if eps == 1:
            for i in range(t):          # the diagonal of M + M^T is 2 M_ii: re-randomise it
                X[i, i] = F(rng.randrange(F.order()))
        if X.is_invertible():
            return X


def random_full_rank(F, k, cols, rng):
    while True:
        G = mat(F, k, cols, [F(rng.randrange(F.order())) for _ in range(k * cols)])
        if G.rank() == k:
            return G


def random_invertible(F, k, rng):
    return random_full_rank(F, k, k, rng)


def random_instance(family, k, m, n, q, eps_a, eps_b, seed=None):
    """Instance from R_{2-sMCE} or R_{sIMCE} (Section 4.3) with independent S1, S2.
    Returns (G0, G1, G2, G3), (S1, S2, A, B) where G3 = S2 G2 Q (2-sMCE) or G3 = G0 (sIMCE)."""
    F, rng = GF(q), random.Random(seed)
    A = random_symmetric_invertible(F, m, eps_a, rng)
    B = random_symmetric_invertible(F, n, eps_b, rng)
    Q = kron(A.transpose(), B)
    S1, S2 = random_invertible(F, k, rng), random_invertible(F, k, rng)
    G0 = random_full_rank(F, k, m * n, rng)
    if family == "2-sMCE":
        G2 = random_full_rank(F, k, m * n, rng)
        return (G0, S1 * G0 * Q, G2, S2 * G2 * Q)
    if family == "sIMCE":
        G2 = S2 * G0 * Q.inverse()           # public key of [KLP26]: (G0, G1, G2); G3 = G0, solution (S1, S2^{-1}, A, B)
        return (G0, S1 * G0 * Q, G2, G0)
    
    raise ValueError(family)


# ----------------------------------------------------------------------------- Algorithm 1
def step1(G0, G1, G2, G3, eps):
    """Basis of the solution space of (2)-(4), as a list of pairs (U1, U2)."""
    F, k = G0.base_ring(), G0.nrows()
    I = identity_matrix(F, k, sparse=_sparse(F))
    Z = mat(F, k * k, k * k)
    # vec(U P) = (I (x) P^T) vec(U);  vec(P U^T) = (P (x) I) vec(U^T) = (P (x) I) K vec(U), K the commutation matrix
    perm = [i * k + j for j in range(k) for i in range(k)]          # (M K)[:, j k + i] = M[:, i k + j]

    def L(P_left, P_right):                                          # equations  U P_left = eps P_right U^T
        return kron(I, P_left.transpose()) - eps * kron(P_right, I).matrix_from_columns(perm)

    L1 = block_matrix(1, 2, [L(G1 * G0.transpose(), G0 * G1.transpose()), Z], subdivide=False)
    L2 = block_matrix(1, 2, [Z, L(G3 * G2.transpose(), G2 * G3.transpose())], subdivide=False)
    L3 = block_matrix(1, 2, [-eps * kron(G2 * G1.transpose(), I).matrix_from_columns(perm),
                             kron(I, (G3 * G0.transpose()).transpose())], subdivide=False)
    K = L1.stack(L2).stack(L3).right_kernel().basis()
    return [(mat(F, k, k, list(v[:k * k])), mat(F, k, k, list(v[k * k:]))) for v in K]


def projections_needed(k, m, n, d, eps_b):
    """t = max(1, ceil((d_{eps_B}(n) - 1) / (2 k n - d m))); None if 2 k n <= d m."""
    if 2 * k * n <= d * m:
        return None
    return max(1, -(-(sym_dimension(n, eps_b) - 1) // (2 * k * n - d * m)))


def step2(G0, G1, G2, G3, Us, m, n, eps_b, t, rng):
    """Projected linear system; returns (kernel basis, DB, t).  Unknown vector:
    [v_{1,1} .. v_{1,d} | ... | v_{t,1} .. v_{t,d} | free entries of Y]."""
    F, k, d = G0.base_ring(), G0.nrows(), len(Us)
    DB = duplication(F, n, eps_b)
    us = [vector(F, [F(rng.randrange(F.order())) for _ in range(m)]) for _ in range(t)]
    pairs = [(G0, [U1 for U1, _ in Us], G1), (G2, [U2 for _, U2 in Us], G3)]
    Ws = [[Uj * Gr for Uj in Ulist] for _, Ulist, Gr in pairs]
    blocks = []
    for i, u in enumerate(us):
        for p, (Gl, _, _) in enumerate(pairs):
            for r in range(k):
                row = [mat(F, n, d * m)] * i                                   # other projections: zero
                row.append(block_matrix(1, d, [reshape(W[r], m, n).transpose() for W in Ws[p]], subdivide=False))
                row += [mat(F, n, d * m)] * (t - 1 - i)
                w = u * reshape(Gl[r], m, n)                                   # u^T D_r, length n
                Yc = mat(F, n, n * n)                                          # (w Y)_b = sum_c w_c Y_{c,b}
                for b in range(n):
                    for c in range(n):
                        Yc[b, c * n + b] = w[c]
                row.append(-Yc * DB)
                blocks.append(block_matrix(1, len(row), row, subdivide=False))
    M = block_matrix(len(blocks), 1, blocks, subdivide=False)
    return M.right_kernel().basis(), DB, t


def step3(G0, G1, Us, lam, Y, m, n, eps_a):
    """(U1, U2) = sum_j lambda_j (U_{1,j}, U_{2,j}); X with X^T W_r = D_r Y; returns (U1, U2, X) or None."""
    F, k, d = G0.base_ring(), G0.nrows(), len(Us)
    U1 = sum(lam[j] * Us[j][0] for j in range(d))
    U2 = sum(lam[j] * Us[j][1] for j in range(d))
    W = U1 * G1
    Wst = block_matrix(1, k, [reshape(W[r], m, n) for r in range(k)], subdivide=False)          # m x kn
    Rst = block_matrix(1, k, [reshape(G0[r], m, n) * Y for r in range(k)], subdivide=False)
    try:
        X = Wst.transpose().solve_right(Rst.transpose())                                        # W^T X = R^T
    except ValueError:
        return None
    if Wst.rank() < m or X.transpose() != eps_a * X:
        return None
    return U1, U2, X


def algorithm1(G0, G1, G2, G3, m, n, eps_a, eps_b, t=None, seed=0):
    """Returns (output, info): output = (S1, S2, A, B) up to scalars, or None;
    info = dict(d, t, kernel, time_step1, time_step2, time_step3)."""
    F, k = G0.base_ring(), G0.nrows()
    rng = random.Random(seed)
    t0 = time.perf_counter()
    Us = step1(G0, G1, G2, G3, eps_a * eps_b)
    d = len(Us)
    t1 = time.perf_counter()
    if t is None:
        t = projections_needed(k, m, n, d, eps_b)
        if t is None:
            return None, dict(d=d, t=None, kernel=None, time_step1=t1 - t0, time_step2=0.0, time_step3=0.0)
    ker, DB, t = step2(G0, G1, G2, G3, Us, m, n, eps_b, t, rng)
    t2 = time.perf_counter()
    out = None
    if len(ker) == 1:
        v = ker[0]
        V = mat(F, t * d, m, list(v[:t * d * m]))                    # row (i, j) = v_{i,j}
        Y = mat(F, n, n, list(DB * v[t * d * m:]))
        i0 = next((i for i in range(t) if not V[i * d:(i + 1) * d, :].is_zero()), None)
        if i0 is not None and Y.is_invertible():
            Vi = V[i0 * d:(i0 + 1) * d, :]
            j0 = next(j for j in range(d) if not Vi[j].is_zero())
            c0 = next(c for c in range(m) if Vi[j0, c] != 0)
            lam = [Vi[j, c0] / Vi[j0, c0] for j in range(d)]
            if all(Vi[j] == lam[j] * Vi[j0] for j in range(d)):
                res = step3(G0, G1, Us, lam, Y, m, n, eps_a)
                if res is not None:
                    U1, U2, X = res
                    if U1.is_invertible() and U2.is_invertible() and X.is_invertible():
                        out = (U1.inverse(), U2.inverse(), X.inverse().transpose(), Y)
    t3 = time.perf_counter()
    return out, dict(d=d, t=t, kernel=len(ker), time_step1=t1 - t0, time_step2=t2 - t1, time_step3=t3 - t2)


def verify(G0, G1, G2, G3, out):
    S1, S2, A, B = out
    Q = kron(A.transpose(), B)
    return bool(G1 == S1 * G0 * Q and G3 == S2 * G2 * Q)


# ----------------------------------------------------------------------------- command line
def main():
    ap = argparse.ArgumentParser(description="Algorithm 1 on a random 2-sMCE / sIMCE instance")
    ap.add_argument("--family", choices=["2-sMCE", "sIMCE"], default="sIMCE")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--q", type=int, default=4093)
    ap.add_argument("--eps-a", type=int, choices=[1, -1], default=1)
    ap.add_argument("--eps-b", type=int, choices=[1, -1], default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    G = random_instance(a.family, a.k, a.m, a.n, a.q, a.eps_a, a.eps_b, a.seed)
    out, info = algorithm1(*G, a.m, a.n, a.eps_a, a.eps_b, seed=a.seed)
    print(f"{a.family} instance k={a.k} m={a.m} n={a.n} q={a.q} (eps_A, eps_B)=({a.eps_a:+d},{a.eps_b:+d})")
    print(f"Step 1: d = {info['d']}   Step 2: t = {info['t']}, kernel dimension = {info['kernel']}")
    print(f"times: step 1 {info['time_step1']:.2f}s, step 2 {info['time_step2']:.2f}s, step 3 {info['time_step3']:.2f}s")
    if out is None:
        print("no solution returned (condition (C1) or (C2) failed)")
        return
    
    ok = verify(*G, out)
    print(f"verified: {ok};")


if __name__ == "__main__":
    main()
