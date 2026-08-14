import numpy as np
import sympy as sp
import scipy.linalg as linalg
import pymittagleffler
import math

_BASIS_CACHE = {}
_EVALUATE_BASIS_CACHE = {}
_MATRIX_TEMPLATE_CACHE = {}

def get_basis(m):
    if m not in _BASIS_CACHE:
        x, nu = sp.symbols('x nu')
        H_basis = []
        for d in range(m + 1):
            for i in range(d, -1, -1):
                j = d - i
                H_basis.append((x**i) * (nu**j))

        _BASIS_CACHE[m] = (H_basis, x, nu)

    return _BASIS_CACHE[m]

def generator(f, x, nu, r, kappa, theta, eta, rho):
    df_dx    = sp.diff(f, x)
    df_dnu   = sp.diff(f, nu)
    d2f_dx2  = sp.diff(f, x, 2)
    d2f_dxnu = sp.diff(f, x, nu)
    d2f_dnu2 = sp.diff(f, nu, 2)
    
    result = (
        (r - 0.5 * nu) * df_dx +
        kappa * (theta - nu) * df_dnu +
        0.5 * nu * d2f_dx2 +
        rho * eta * nu * d2f_dxnu +
        0.5 * eta**2 * nu * d2f_dnu2
    )
    
    return sp.expand(result)

def build_generator_matrix(m, r, kappa, theta, eta, rho):
    if m not in _MATRIX_TEMPLATE_CACHE:
        H_basis, x_sym, nu_sym = get_basis(m)
        r_sym, kappa_sym, theta_sym, eta_sym, rho_sym = sp.symbols('r kappa theta eta rho')
        A_sym = sp.Matrix(np.zeros((len(H_basis), len(H_basis))))

        for j in range(len(H_basis)):
            func = H_basis[j]
            result = generator(func, x_sym, nu_sym, r_sym, kappa_sym, theta_sym, eta_sym, rho_sym)
            poly = sp.Poly(result, x_sym, nu_sym)

            for i in range(len(H_basis)):
                basis_element = H_basis[i]
                A_sym[i, j] = poly.coeff_monomial(basis_element)

        matrix_func = sp.lambdify((r_sym, kappa_sym, theta_sym, eta_sym, rho_sym), A_sym, "numpy")
        _MATRIX_TEMPLATE_CACHE[m] = matrix_func

    matrix_func = _MATRIX_TEMPLATE_CACHE[m]
    A = np.array(matrix_func(r, kappa, theta, eta, rho))
    return A

def get_evaluate_basis(m):
    if m not in _EVALUATE_BASIS_CACHE:
        H_basis, x, nu = get_basis(m)
        evaluate_basis = sp.lambdify((x, nu), H_basis, "numpy")

        p = np.zeros(len(H_basis))
        x = sp.symbols("x")
        p_idx = H_basis.index(x**m)
        p[p_idx] = 1.0

        _EVALUATE_BASIS_CACHE[m] = (evaluate_basis, p)
        
    return _EVALUATE_BASIS_CACHE[m]

def sort_schur(M):
    Z, Q = linalg.schur(M)
    n = Z.shape[0]
    for i in range(n):
        for j in range(0, n - i - 1):
            if Z[j, j] < Z[j + 1, j + 1]:
                Z, Q, _ = linalg.lapack.dtrexc(Z, Q, j + 1, j + 2)
    return Z, Q

def get_block_sizes(Z, tol=1e-8):
    diag = np.diag(Z)
    blocks = []
    current_size = 1
    for i in range(len(diag) - 1):
        if abs(diag[i] - diag[i+1]) < tol:
            current_size += 1
        else:
            blocks.append(current_size)
            current_size = 1
    blocks.append(current_size)
    return blocks

def compute_c_coefficients(k, alpha, beta=1.0):
    c = np.zeros((k + 1, k + 1))

    # j = k
    for i in range(k + 1):
        c[i, i] = 1
    
    # j = 0
    for i in range(1, k + 1):
        c[0, i] = (1 - beta - alpha * (i - 1)) * c[0, i - 1]

    # j = 1, ..., k - 1
    for j in range(1, k):
        for i in range(j + 1, k + 1):
            c[j, i] = c[j - 1, i - 1] + (1 - beta - alpha * (i - 1) + j) * c[j, i - 1]
    
    return c[:, k]

def ml_derivative(z, k, alpha, beta=1.0):
    if abs(z) < 1e-15:
        return math.factorial(k) / math.gamma(alpha * k + beta)

    if k == 0:
        return pymittagleffler.mittag_leffler(z, alpha, beta).real
    
    c = compute_c_coefficients(k, alpha, beta)
    result = 0.0
    for j in range(k + 1):
        result += c[j] * pymittagleffler.mittag_leffler(z, alpha, alpha * k + beta - j).real
    
    result = result / (alpha ** k)
    return result

def compute_diagonal_blocks(Z, block_sizes, alpha, beta=1.0):
    F = np.zeros((Z.shape[0], Z.shape[0]))

    start_idx = 0
    for bs in block_sizes:
        end_idx = start_idx + bs

        Z_ii = Z[start_idx:end_idx, start_idx:end_idx]
        lambda_i = Z[start_idx, start_idx]

        M_i = Z_ii - lambda_i * np.eye(bs)
        F_ii = np.zeros((bs, bs))
        M_k = np.eye(bs)
        for k in range(bs):
            F_ii += (ml_derivative(lambda_i, k, alpha, beta) / math.factorial(k)) * M_k
            M_k = M_k @ M_i

        F[start_idx:end_idx, start_idx:end_idx] = F_ii
        start_idx = end_idx

    return F

def compute_offdiagonal_blocks(Z, F, block_sizes):
    start_id = []
    curr = 0
    for bs in block_sizes:
        start_id.append(curr)
        curr += bs
    start_id.append(curr)

    for i in range(len(block_sizes) - 2, -1, -1):
      for j in range(i + 1, len(block_sizes)):
          slice_i = slice(start_id[i], start_id[i+1])
          slice_j = slice(start_id[j], start_id[j+1])

          Z_ii = Z[slice_i, slice_i]
          Z_jj = Z[slice_j, slice_j]

          Z_ij = Z[slice_i, slice_j]
          F_ii = F[slice_i, slice_i]
          F_jj = F[slice_j, slice_j]
          R_ij = F_ii @ Z_ij - Z_ij @ F_jj
          for k in range(i + 1, j):
              slice_k = slice(start_id[k], start_id[k+1])
              F_ik = F[slice_i, slice_k]
              Z_kj = Z[slice_k, slice_j]
              Z_ik = Z[slice_i, slice_k]
              F_kj = F[slice_k, slice_j]
              R_ij += F_ik @ Z_kj - Z_ik @ F_kj

          F[slice_i, slice_j] = linalg.solve_sylvester(Z_ii, -Z_jj, R_ij)
            
    return F

def ml_moment(m, kappa, theta, eta, rho, v0, T, alpha):
    A = build_generator_matrix(m, 0.0, kappa, theta, eta, rho)
    
    # T^{alpha} * A = Q * Z * Q^T
    Z, Q = sort_schur(T**(alpha) * A)
    block_sizes = get_block_sizes(Z)
    
    F_T_diag = compute_diagonal_blocks(Z, block_sizes, alpha)
    F_T = compute_offdiagonal_blocks(Z, F_T_diag, block_sizes)
    E_alpha = Q @ F_T @ Q.T
    
    evaluate_basis, p = get_evaluate_basis(m)

    H_vector = np.array(evaluate_basis(0.0, v0))

    moment = H_vector.T @ E_alpha @ p
    return moment