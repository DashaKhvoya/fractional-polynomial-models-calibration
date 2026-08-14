import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import multivariate_normal
from MomentsCalculation import fukasawa, moments_mittag_leffler, svi_calibration

def generate_probabilistic_population(bounds, popsize):
    num_params = len(bounds)
    num_candidates = popsize * num_params
    population = []

    alpha_cov = 0.1
    cov_matrix = np.diag([alpha_cov * (b[1] - b[0])**2 for b in bounds])
    population.append(np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds]))
    while len(population) < num_candidates:
        candidates = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (200, num_params))
        probs = np.zeros(200)

        for start_x in population:
            probs += multivariate_normal.pdf(candidates, start_x, cov_matrix)

        population.append(candidates[np.argmin(probs)])

    return np.array(population)
    
def objective(params, T_array, moments, fukasawa_moments, weights_dict):
    kappa, theta, eta, rho, v0, alpha = params

    total_loss = 0.0
    epsilon = 5e-4

    for T in T_array:
        for m in moments:
            f_m = fukasawa_moments[(T, m)]
            w_Tm = weights_dict[(T, m)]

            try:
                ml_m = moments_mittag_leffler.ml_moment(m, kappa, theta, eta, rho, v0, T, alpha)
                total_loss += w_Tm * (((f_m - ml_m) / (abs(f_m) + epsilon))**2)
            except Exception:
                return 1e9

    return total_loss

def compute_fukasawa_moments(T_array, moments, smiles_dict, k_grid_dict, S0 = 100.0):
    fukasawa_results = {}

    for T in T_array:
        k_grid = k_grid_dict[T]
        ivs = smiles_dict[T]

        svi_params = svi_calibration.calibrate_svi(k_grid, ivs, T, S0)

        for m in moments:
            f_m = fukasawa.fukasawa(svi_params, m)
            fukasawa_results[(T, m)] = f_m

    return fukasawa_results

def calibrate_fractional_heston(T_array, moments, smiles_dict, k_grid_dict, weights_dict, S0 = 100.0):
    fukasawa_moments = compute_fukasawa_moments(T_array, moments, smiles_dict, k_grid_dict, S0)

    bounds = [
        (1e-3, 10.0),  # kappa
        (1e-4, 0.5),   # theta
        (1e-3, 5.0),   # eta
        (-0.99, 0.0),  # rho
        (1e-4, 1.0),   # v0
        (0.1, 1.0)     # alpha
    ]

    custom_population = generate_probabilistic_population(bounds, popsize=18)

    result = differential_evolution(
        objective,
        bounds,
        args=(T_array, moments, fukasawa_moments, weights_dict),
        popsize=18,
        init=custom_population,
        mutation=0.9,
        recombination=0.5,
        polish=True,
        strategy='best1bin',
        workers=1,
        disp=True)

    return result.x