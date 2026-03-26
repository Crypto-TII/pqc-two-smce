import gc
import sys
import argparse
from random import choice
import numpy as np
from time import time
    
# SageMath imports
from sage.all import (
    matrix,
    vector,
    zero_matrix,
    FiniteField,
    identity_matrix,
)

def arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-k", "--code_dimension", type=int, help="Code dimension parameter k", required=False)
    parser.add_argument("-m", "--code_length_m", type=int, help="Code length parameter m", required=False)
    parser.add_argument("-n", "--code_length_n", type=int, help="Code length parameter n", required=True)
    parser.add_argument("-q", "--prime", type=int, help="Field characteristic", required=True)
    parser.add_argument("-i", "--simce", help="For solving sIMCE instances", action="store_true")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    options = parser.parse_args(args)
    return options


def sample_symmetric_matrix(d, q, skew=False):
    sign = {True: -1, False: 1}[skew]
    if d % 2:
        sign = 1
    
    F = FiniteField(q)
    l = (d * (d + 1) // 2)

    stop = False
    while not stop:

        mij = [F.random_element() for _ in range(0, l, 1)]

        M = zero_matrix(F, d, d)
        o = -1
        for i in range(0, d, 1):
            for j in range(i, d, 1):
                o += 1
                M[i, j] = int(mij[o])

        M += sign * M.transpose()

        stop = M.is_invertible()

    assert(M.is_symmetric() or M.is_skew_symmetric())
    assert(M.is_invertible())
    return M


def sample_full_rank_matrix(rows, cols, q):
    F = FiniteField(q)
    l = rows * cols

    stop = False
    while not stop:
        m = [F.random_element() for _ in range(0, l, 1)]
        M = matrix(F, rows, cols, [int(mij) for mij in m])
        stop = (M.rank() == rows)
    
    assert(M.rank() == rows)
    return M


def generate_instance(k, m, n, q, simce=False):
    skew_a = choice([True, False])
    skew_b = choice([True, False])
    
    # Secret matrices
    S = sample_full_rank_matrix(k, k, q)
    A = sample_symmetric_matrix(m, q, skew=skew_a)
    B = sample_symmetric_matrix(n, q, skew=skew_b)
    Q = A.transpose().tensor_product(B, subdivide=False)
    
    # Public matrices
    G0 = sample_full_rank_matrix(k, m*n, q)
    G1 = S * G0 * Q
    if simce:
        print(f'sIMCE instance')
        G2 = S.inverse() * G0 * Q.inverse()
        G3 = G0
    else:
        print(f'2-sMCE instance')
        G2 = sample_full_rank_matrix(k, m*n, q)
        G3 = S * G2 * Q
    
    return (G0, G1), (G2, G3)


def commutation_matrix(m, n):
    perm_indices = np.arange(m * n)
    perm_indices = perm_indices.reshape((m, n), order='F')
    perm_indices = perm_indices.T.ravel(order='F')
    identity_matrix = np.eye(m * n)
    K = identity_matrix[perm_indices, :]
    return K


def solver_two_smce(F, G0, G1, G2, G3):
    start = time()
    print(f'\nWe first recover the invertible matrix S')
    
    Ik = identity_matrix(F, k, k)
    K = matrix(F, commutation_matrix(k, k))

    # Equations determined by (G₀G₁ᵀ, G₁G₀ᵀ) and (G₂G₃ᵀ, G₃G₂ᵀ)
    G10 = G1 * G0.transpose()
    G32 = G3 * G2.transpose()

    assert(G10.rank() == k)
    assert(G32.rank() == k)
    X10 = Ik.tensor_product(G10)
    Y10 = G10.tensor_product(Ik)
    Y10 *= K

    X32 = Ik.tensor_product(G32)
    Y32 = G32.tensor_product(Ik)
    Y32 *= K

    # Sanity checks
    vec = lambda m: vector(m.list())
    assert(vec(G10.transpose()) != vec(G10))
    assert(K * vec(G10.transpose()) == vec(G10))
    assert(vec(G32.transpose()) != vec(G32))
    assert(K * vec(G32.transpose()) == vec(G32))

    # Equations determined by (G₀G₃ᵀ, G₁G₂ᵀ) and (G₂G₁ᵀ, G₃G₀ᵀ)
    G30 = G3 * G0.transpose()
    G12 = G1 * G2.transpose()

    assert(G30.rank() == k)
    assert(G12.rank() == k)
    X30 = Ik.tensor_product(G30)
    Y12 = G12.tensor_product(Ik)
    Y12 *= K

    X12 = Ik.tensor_product(G12)
    Y30 = G30.tensor_product(Ik)
    Y30 *= K

    # Sanity checks
    vec = lambda m: vector(m.list())
    assert(vec(G30.transpose()) != vec(G10))
    assert(K * vec(G30.transpose()) == vec(G30))
    assert(vec(G12.transpose()) != vec(G12))
    assert(K * vec(G12.transpose()) == vec(G12))

    for epsilon in [+1,-1]:
        #      Symmetric structure gives rank k(k-1)/2
        # Skew-symmetric structure gives rank k(k+1)/2
        # Build initial system
        current_system = zero_matrix(F, 0, k**2)
        current_system = current_system.stack(X10 - epsilon*Y10)
        current_system = current_system.stack(X32 - epsilon*Y32)
        current_system = current_system.stack(X30 - epsilon*Y12)
        current_system = current_system.stack(X12 - epsilon*Y30)

        kernel = current_system.right_kernel().matrix()
        del current_system
        gc.collect()
        if kernel.nrows() == 1:
            break

    assert(kernel.nrows() == 1)
    S = matrix(k ,k, kernel[0])
    assert(S.is_invertible())

    print(f'We next recover the (skew) symmetric matrices A and B')

    G1_ = S.inverse() * G1
    # G3_ = S_.inverse() * G3

    num_X = m*m
    num_Y = n*n
    num_vars = num_X + num_Y

    # map variables to vector indices
    def idx_A(i,j):
        return i*m + j

    def idx_B(i,j):
        return num_X + i*n + j

    reduced_system = matrix(F, 0, num_vars)
    current_rank = 0
    for r in range(k):
        for c in range(m*n):

            sys.stdout.write(f"\rBuilding reduced system: {current_rank}/{num_vars}")
            sys.stdout.flush()
            if current_rank == (num_vars - 1):
                break

            row = zero_matrix(F, 1, num_vars)

            # Contributions from A
            for i in range(m):
                for j in range(m):
                    for t in range(n):
                        col_index = j*n + t
                        if col_index == c:
                            coeff = G1_[r, i*n + t]
                            row[0, idx_A(i,j)] += coeff
            # Contributions from B
            for a in range(n):
                for b in range(n):
                    for t in range(m):
                        col_index = t*n + b
                        if col_index == c:
                            coeff = -G0[r, t*n + a]
                            row[0, idx_B(a,b)] += coeff

            if reduced_system.stack(row).rank() == (current_rank + 1):
                reduced_system = reduced_system.stack(row)
                current_rank += 1
            
    print('\n')
    print(f'{reduced_system = }\n{reduced_system.rank() = }')
    solutions = reduced_system.right_kernel().matrix()
    del reduced_system
    gc.collect()
    assert(solutions.nrows() == 1)

    solutions = solutions[0]
    A = matrix(F,m,m,solutions[:num_X])
    B = matrix(F,n,n,solutions[num_X:])
    assert(A.is_invertible())
    assert(B.is_invertible())
    A = A.inverse()

    Q = A.tensor_product(B)
    assert(G1 == (S * G0 * Q))
    assert(G3 == (S * G2 * Q))

    end = time()
    elapsed_time = end - start

    print(f'\n{elapsed_time = }\n')

    return S, (A, B)

def main(k, m, n, q, simce=False):
    F = FiniteField(q)

    (G0, G1), (G2, G3) = generate_instance(k, m, n, q, simce=simce)
    
    print(f'\nPublic codes:')
    print(f'\nG₀:\n{G0}')
    print(f'\nG₁:\n{G1}')
    print(f'\nG₂:\n{G2}')
    print(f'\nG₃:\n{G3}')

    S, (A, B) = solver_two_smce(F, G0, G1, G2, G3)

    print('We successfully found an invertible matrix S along with a pair of (skew) symmetric matrices (A,B) that maps G₀ to G₁ and G₂ to G₃\n')
    print(f'S:\n{S}\n')
    print(f'A:\n{A}\n')
    print(f'B:\n{B}\n')

    return True


if __name__ == '__main__':

    k = arguments(sys.argv[1:]).code_dimension
    n = arguments(sys.argv[1:]).code_length_n
    m = arguments(sys.argv[1:]).code_length_m
    q = arguments(sys.argv[1:]).prime
    
    print(f'\nCode dimension, k: {k}')
    print(f'Code lenght,  m•n: {m}•{n} = {m*n}')
    print(f'Field size,     q: {q}\n')
    
    simce = arguments(sys.argv[1:]).simce    
    correct = main(k, m, n, q, simce=simce)
    assert(correct)