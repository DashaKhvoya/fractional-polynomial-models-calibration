import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

def raw_to_jw(a, b, rho, m, sigma, T):
    v_t = (a + b * (-rho * m + np.sqrt(m**2 + sigma**2))) / T
    w_t = v_t * T
    psi_t = (1.0 / np.sqrt(w_t)) * (b / 2.0) * (rho - m / np.sqrt(m**2 + sigma**2))
    p_t = (1.0 / np.sqrt(w_t)) * b * (1.0 - rho)
    c_t = (1.0 / np.sqrt(w_t)) * b * (1.0 + rho)
    v_hat_t = 1.0 / T * (a + b * sigma * np.sqrt(1.0 - rho**2))
    return v_t, psi_t, p_t, c_t, v_hat_t

def jw_to_raw(v_t, psi_t, p_t, c_t, v_hat_t, T):
    w_t = v_t * T
    b = np.sqrt(w_t) / 2.0 * (c_t + p_t)
    rho = 1.0 - p_t * np.sqrt(w_t) / b
    rho = np.clip(rho, -0.999999, 0.999999)
    
    beta = rho - (2.0 * psi_t * np.sqrt(w_t)) / b
    beta = np.clip(beta, -0.999999, 0.999999)
    alpha = np.sign(beta) * np.sqrt(max(1.0 / beta**2 - 1.0, 0.0))
    if abs(alpha) < 1e-15:
        m = 0.0
        sigma = ((v_t - v_hat_t) * T) / (b * (1.0 - np.sqrt(1.0 - rho**2)))
    else:
        denom = b * (-rho + np.sign(alpha) * np.sqrt(1.0 + alpha**2) - alpha * np.sqrt(1.0 - rho**2))
        m = ((v_t - v_hat_t) * T) / denom if abs(denom) > 1e-15 else 0.0
        sigma = alpha * m
    
    a = v_hat_t * T - b * sigma * np.sqrt(1.0 - rho**2)
    return a, b, rho, m, sigma

def durrleman_condition(a, b, rho, m, sigma, x):
    w = a + b * (rho * (x - m) + np.sqrt((x - m)**2 + sigma**2))
    w = np.maximum(w, 1e-9)
    
    dw = b * (rho + (x - m) / np.sqrt((x - m)**2 + sigma**2))
    d2w = (b * sigma**2) / ((x - m)**2 + sigma**2)**(3/2)
    
    durr_cond = (1.0 - x / (2.0 * w) * dw)**2 - (dw**2 / 4.0) * (1.0 / w + 1.0 / 4.0) + d2w / 2.0
    return np.min(durr_cond) >= 0.0

def find_inner_params(x, w, sigma, m, weights):
    y = (x - m) / sigma
    z = np.sqrt(y**2 + 1.0)
    # (A^T @ A) @ params = w @ A
    A = np.column_stack((np.ones_like(y), y, z))
    try:
        a, d, c = np.linalg.solve((A.T * weights) @ A, (A.T * weights) @ w)
        
        # Check if (a,d,c) \in D
        if (c >= 0.0 and c <= 4.0 * sigma and
            abs(d) <= c and abs(d) <= 4.0 * sigma - c and
            a >= 0.0 and a <= max(w)):
            b = c / sigma
            rho = d / c if c > 1e-9 else 0.0
            rho = np.clip(rho, -1.0, 1.0)
            return a, b, rho, np.sum(weights * (A @ np.array([a, d, c]) - w)**2)
    except np.linalg.LinAlgError:
        pass
    
    # Find min of f on boundary(D)
    def objective(params):
        a_p, d_p, c_p = params
        return np.sum(weights * (A @ np.array([a_p, d_p, c_p]) - w)**2)
    
    D = [
        {"type": "ineq", "fun": lambda x_p: x_p[2]},                          # c >= 0
        {"type": "ineq", "fun": lambda x_p: 4.0 * sigma - x_p[2]},            # c <= 4*sigma
        {"type": "ineq", "fun": lambda x_p: x_p[2] - x_p[1]},                 # c >= d
        {"type": "ineq", "fun": lambda x_p: x_p[2] + x_p[1]},                 # c >= -d
        {"type": "ineq", "fun": lambda x_p: 4.0 * sigma - x_p[2] - x_p[1]},   # 4*sigma - c >= d
        {"type": "ineq", "fun": lambda x_p: 4.0 * sigma - x_p[2] + x_p[1]},   # 4*sigma - c >= -d
        {"type": "ineq", "fun": lambda x_p: x_p[0]},                          # a >= 0
        {"type": "ineq", "fun": lambda x_p: max(w) - x_p[0]}                  # a <= max(w)
    ]

    x0 = np.array([np.mean(w), 0.0, 0.1 * sigma])

    result = minimize(
        objective, 
        x0, 
        method='SLSQP',
        constraints=D
    )
    a, d, c = result.x
    b = c / sigma
    rho = d / c if c > 1e-9 else 0.0
    rho = np.clip(rho, -1.0, 1.0)
    return a, b, rho, result.fun

def calibrate_svi(k_grid, implied_vols, T, S0):
    valid = ~np.isnan(implied_vols) & (implied_vols > 1e-4)
    k_grid_f = k_grid[valid]
    implied_vols_f = implied_vols[valid]
    # total var = (implied vol)^2 * T
    w = (implied_vols_f**2) * T

    vega = S0 * np.sqrt(T) * norm.pdf(-k_grid_f / np.sqrt(w) + np.sqrt(w) / 2.0)
    weights = vega / np.sum(vega)
    
    # Choosing start sigma and m
    start_points = []
    num_restarts = 10
    num_candidates = 100
    num_params = 2
    bounds = [(1e-4, 1.0), (min(k_grid_f), max(k_grid_f))]
    alpha_cov = 0.1
    cov_matrix = np.diag([alpha_cov * (b[1] - b[0])**2 for b in bounds])
    min_f = np.inf
    min_params = [0.1, 0.0]

    def objective_outer(params):
        sigma_p, m_p = params
        _, _, _, result = find_inner_params(k_grid_f, w, sigma_p, m_p, weights)
        return result

    for restart in range(num_restarts):
        candidates = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (num_candidates, num_params))
        if len(start_points) == 0:
            start_points.append(candidates[0])
        else:
            probabilities = []
            for c_cand in candidates:
                prob_c = 0.0
                for start_x in start_points:
                    prob_c += multivariate_normal.pdf(c_cand, start_x, cov_matrix)
                probabilities.append(prob_c / len(start_points))
            
            start_points.append(candidates[np.argmin(probabilities)])
        
        sigma_cand, m_cand = start_points[-1]
        f_val = objective_outer([sigma_cand, m_cand])
        if f_val < min_f:
            min_f = f_val
            min_params = [sigma_cand, m_cand]
    
    res_nelder_mead = minimize(objective_outer, min_params, method="Nelder-Mead", bounds=bounds)
    sigma_opt, m_opt = res_nelder_mead.x
    a_opt, b_opt, rho_opt, _ = find_inner_params(k_grid_f, w, sigma_opt, m_opt, weights)
    # Durrleman’s condition check
    k_wide = np.linspace(-5.0, 5.0, 200)
    if durrleman_condition(a_opt, b_opt, rho_opt, m_opt, sigma_opt, k_wide):
        return a_opt, b_opt, rho_opt, m_opt, sigma_opt
    
    # Durrleman’s condition failed -> Arbitrage elimination step
    v_t, psi_t, p_t, c_t, v_hat_t = raw_to_jw(a_opt, b_opt, rho_opt, m_opt, sigma_opt, T)
    c_t_new = p_t + 2.0 * psi_t
    v_hat_t_new = v_t * ((4.0 * p_t * c_t_new) / (p_t + c_t_new)**2)

    def objective_arb(params):
        c_val, v_hat_val = params
        a_new, b_new, rho_new, m_new, sigma_new = jw_to_raw(v_t, psi_t, p_t, c_val, v_hat_val, T)
        
        if not durrleman_condition(a_new, b_new, rho_new, m_new, sigma_new, k_wide):
            return 1e9
            
        w_new = a_new + b_new * (rho_new * (k_grid_f - m_new) + np.sqrt((k_grid_f - m_new)**2 + sigma_new**2))
        w_old = a_opt + b_opt * (rho_opt * (k_grid_f - m_opt) + np.sqrt((k_grid_f - m_opt)**2 + sigma_opt**2))
        return np.sum(weights * (w_old - w_new)**2)

    c_bounds = (min(c_t, c_t_new), max(c_t, c_t_new))
    v_bounds = (min(v_hat_t, v_hat_t_new), max(v_hat_t, v_hat_t_new))
    result_arb = minimize(objective_arb, [c_t_new, v_hat_t_new], method="Nelder-Mead", bounds=[c_bounds, v_bounds])
    
    c_t_res, v_hat_t_res = result_arb.x
    return jw_to_raw(v_t, psi_t, p_t, c_t_res, v_hat_t_res, T)