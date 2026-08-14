import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from HestonModel import heston_fractional

def build_volatility_smile(S0, v0, kappa, theta, eta, rho, r, T_array, dt, ds, alpha, n_paths):
    is_scalar = np.isscalar(T_array) or isinstance(T_array, (float, int))
    maturities = [float(T_array)] if is_scalar else [float(t) for t in T_array]
    T_max = max(maturities)

    paths_matrix = []
    t_grid = None

    for _ in range(n_paths):
        t_grid, _, S_path, _ = heston_fractional.single_path_fractional_heston(
            S0, v0, kappa, theta, eta, rho, r, T_max, dt, ds, alpha
        )
        paths_matrix.append(S_path)

    paths_matrix = np.array(paths_matrix)

    all_smiles = []
    all_rse = []
    all_k_grids = []
    all_moments = []
    all_moments_se = []

    k_grid_bounds = {
        0.25: 0.5, 0.50: 0.6, 0.75: 0.7, 1.00: 0.8, 
        1.25: 0.9, 1.50: 1.0, 1.75: 1.1, 2.00: 1.2}

    for T in maturities:
        idx_T = np.argmin(np.abs(t_grid - T))
        S_T_paths = paths_matrix[:, idx_T].copy()

        F = S0 * np.exp(r * T)
        F_forward = np.mean(S_T_paths)

        # Martingale normalization
        if F_forward > 0:
            S_T_paths = S_T_paths * (F / F_forward)

        mc_moments = {}
        mc_moments_se = {}
        for m in [1, 2, 3, 4]:
            X_m = np.log(S_T_paths / F) ** m
            mc_moments[m] = np.mean(X_m)
            mc_moments_se[m] = np.std(X_m) / np.sqrt(len(S_T_paths))

        bound = k_grid_bounds[round(T, 2)]
        k_grid = np.linspace(-bound, bound, 51)

        implied_vols = []
        rse_list = []

        for k in k_grid:
            K = F * np.exp(k)

            if k >= 0:
                # k >= 0: OTM Call
                payoffs = np.maximum(S_T_paths - K, 0.0)
                mc_forward_price = np.mean(payoffs)
                bs_fun = lambda vol: F * (norm.cdf(-k/vol + vol/2) - np.exp(k) * norm.cdf(-k/vol - vol/2))
            else:
                # k < 0: OTM Put
                payoffs = np.maximum(K - S_T_paths, 0.0)
                mc_forward_price = np.mean(payoffs)
                bs_fun = lambda vol: F * (np.exp(k) * norm.cdf(k/vol + vol/2) - norm.cdf(k/vol - vol/2))

            # RSE calculation
            mean_val = np.mean(payoffs)
            std_val = np.std(payoffs)
            if mean_val > 1e-9:
                rse = (std_val / np.sqrt(n_paths)) / mean_val
            else:
                rse = np.inf

            rse_list.append(rse)

            fun = lambda vol: bs_fun(vol) - mc_forward_price

            try:
                total_vol = brentq(fun, 0.0001, 4.0)
                iv = total_vol / np.sqrt(T)
                implied_vols.append(iv)
            except (ValueError, RuntimeError):
                implied_vols.append(np.nan)

        all_smiles.append(implied_vols)
        all_rse.append(rse_list)
        all_k_grids.append(k_grid)
        all_moments.append(mc_moments)
        all_moments_se.append(mc_moments_se)

    if is_scalar:
        return np.array(all_smiles[0]), np.array(all_rse[0]), np.array(all_k_grids[0]), all_moments[0], all_moments_se[0]
    return np.array(all_smiles), np.array(all_rse), np.array(all_k_grids), all_moments, all_moments_se