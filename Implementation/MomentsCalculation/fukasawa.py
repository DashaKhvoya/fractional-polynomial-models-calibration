import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline

def find_bounds(h_func, f_grid):
    k_left = -1.0
    k_right = 1.0
    N = 40
    ratio_left = 100.0
    ratio_right = 100.0

    while(ratio_left > 0.001 or ratio_right > 0.001):
        dk = (k_right - k_left) / (N - 1)
        k_nodes = np.linspace(k_left, k_right, N)
        
        z = f_grid(k_nodes)
        y = h_func(k_nodes)
        
        idx = np.argsort(z)
        z_s, y_s = z[idx], y[idx]
        
        spline = CubicSpline(z_s, y_s, bc_type='natural')
        
        I = spline(z_s) * norm.pdf(z_s)
        I_max = np.max(np.abs(I))
        
        ratio_left = np.abs(I[0]) / I_max
        ratio_right = np.abs(I[-1]) / I_max
        
        if ratio_left > 0.001:
            k_left -= dk
        if ratio_right > 0.001:
            k_right += dk
            
    return k_left, k_right, spline

def find_spline(h_func, f_grid, k_left, k_right):
    N = 40
    k_nodes = np.linspace(k_left, k_right, N)
        
    z = f_grid(k_nodes)
    y = h_func(k_nodes)
        
    idx = np.argsort(z)
    z_s, y_s = z[idx], y[idx]
    
    spline = CubicSpline(z_s, y_s, bc_type='natural')
        
    return spline

def true_coeff(z_int, c):
    z_left = z_int[:-1]

    A = c[0]
    B = c[1] - 3 * c[0] * z_left
    C = c[2] - 2 * c[1] * z_left + 3 * c[0] * z_left**2
    D = c[3] - c[2] * z_left + c[1] * z_left**2 - c[0] * z_left**3
    return np.array([A, B, C, D])

def find_integral(z_int, c):
    I = 0
    for j in range(len(z_int) - 1):
        phi_j = norm.pdf(z_int[j])
        phi_j_next = norm.pdf(z_int[j + 1])
        
        Phi_j = norm.cdf(z_int[j])
        Phi_j_next = norm.cdf(z_int[j + 1])

        I_j = (
            c[3, j] * (Phi_j_next - Phi_j) +
            c[2, j] * (phi_j - phi_j_next) +
            c[1, j] * (z_int[j] * phi_j - z_int[j + 1] * phi_j_next + Phi_j_next - Phi_j) +
            c[0, j] * (z_int[j]**2 * phi_j - z_int[j + 1]**2 * phi_j_next + 2 * (phi_j - phi_j_next))
        )

        I += I_j
    return I

def fukasawa(svi_params, m):
    def svi_total_val(k):
        a, b, rho, m_svi, sigma_svi = svi_params
        return np.sqrt(a + b * (rho * (k - m_svi) + np.sqrt((k - m_svi)**2 + sigma_svi**2)))
    
    f1 = lambda k: k / svi_total_val(k) - svi_total_val(k) / 2
    f2 = lambda k: k / svi_total_val(k) + svi_total_val(k) / 2

    # Find bounds (for m = 4)
    h1_m4 = lambda x: 4 * x**3 * np.exp(-x)
    h2_m4 = lambda x: x**4 - 4 * x**3

    k_left_h1, k_right_h1, spline_h1 = find_bounds(h1_m4, f1)
    k_left_h2, k_right_h2, spline_h2 = find_bounds(h2_m4, f2)

    k_left = min(k_left_h1, k_left_h2)
    k_right = max(k_right_h1, k_right_h2)
    
    h1 = lambda x: m * x**(m - 1) * np.exp(-x)
    h2 = lambda x: x**m - m * x**(m - 1)
    
    spline_h1 = find_spline(h1, f1, k_left, k_right)
    spline_h2 = find_spline(h2, f2, k_left, k_right)

    coeff_h1 = true_coeff(spline_h1.x, spline_h1.c)
    coeff_h2 = true_coeff(spline_h2.x, spline_h2.c)

    I = find_integral(spline_h1.x, coeff_h1) + find_integral(spline_h2.x, coeff_h2)

    return I
    
