import os
import numpy as np
import time
import volatility_smile

# Create output folder for dataset
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../SimulatedData"))
os.makedirs(output_dir, exist_ok=True)

# Common base parameters
S0 = 100.0
r = 0.02
dt = 1/252
ds = 0.002
n_paths = 100000

# 8 maturities (3:24 months)
T_array = [0.25, 0.50, 0.75, 1.00, 1.25, 1.5, 1.75, 2.0]

# 24 experiments
experiments = [
    {"id": "exp_01_vanilla_market_alpha=1.0", "kappa": 1.15, "theta": 0.04, "eta": 0.2, "rho": -0.4, "v0": 0.04, "alpha": 1.0},
    {"id": "exp_02_vanilla_market_alpha=0.9", "kappa": 1.15, "theta": 0.04, "eta": 0.2, "rho": -0.4, "v0": 0.04, "alpha": 0.9},
    {"id": "exp_03_vanilla_market_alpha=0.8", "kappa": 1.15, "theta": 0.04, "eta": 0.2, "rho": -0.4, "v0": 0.04, "alpha": 0.8},
    {"id": "exp_04_vanilla_market_alpha=0.7", "kappa": 1.15, "theta": 0.04, "eta": 0.2, "rho": -0.4, "v0": 0.04, "alpha": 0.7},
    {"id": "exp_05_vanilla_market_alpha=0.6", "kappa": 1.15, "theta": 0.04, "eta": 0.2, "rho": -0.4, "v0": 0.04, "alpha": 0.6},
    {"id": "exp_06_vanilla_market_alpha=0.5", "kappa": 1.15, "theta": 0.04, "eta": 0.2, "rho": -0.4, "v0": 0.04, "alpha": 0.5},
    {"id": "exp_07_vanilla_market_alpha=0.4", "kappa": 1.15, "theta": 0.04, "eta": 0.2, "rho": -0.4, "v0": 0.04, "alpha": 0.4},
    {"id": "exp_08_vanilla_market_alpha=0.3", "kappa": 1.15, "theta": 0.04, "eta": 0.2, "rho": -0.4, "v0": 0.04, "alpha": 0.3},
    {"id": "exp_09_leverage_effect_alpha=1.0", "kappa":  3.0, "theta": 0.15, "eta": 0.2, "rho": -0.8, "v0":  0.4, "alpha": 1.0},
    {"id": "exp_10_leverage_effect_alpha=0.9", "kappa":  3.0, "theta": 0.15, "eta": 0.2, "rho": -0.8, "v0":  0.4, "alpha": 0.9},
    {"id": "exp_11_leverage_effect_alpha=0.8", "kappa":  3.0, "theta": 0.15, "eta": 0.2, "rho": -0.8, "v0":  0.4, "alpha": 0.8},
    {"id": "exp_12_leverage_effect_alpha=0.7", "kappa":  3.0, "theta": 0.15, "eta": 0.2, "rho": -0.8, "v0":  0.4, "alpha": 0.7},
    {"id": "exp_13_leverage_effect_alpha=0.6", "kappa":  3.0, "theta": 0.15, "eta": 0.2, "rho": -0.8, "v0":  0.4, "alpha": 0.6},
    {"id": "exp_14_leverage_effect_alpha=0.5", "kappa":  3.0, "theta": 0.15, "eta": 0.2, "rho": -0.8, "v0":  0.4, "alpha": 0.5},
    {"id": "exp_15_leverage_effect_alpha=0.4", "kappa":  3.0, "theta": 0.15, "eta": 0.2, "rho": -0.8, "v0":  0.4, "alpha": 0.4},
    {"id": "exp_16_leverage_effect_alpha=0.3", "kappa":  3.0, "theta": 0.15, "eta": 0.2, "rho": -0.8, "v0":  0.4, "alpha": 0.3},
    {"id": "exp_17_Feller_fail_alpha=1.0", "kappa": 6.5482, "theta": 0.0731, "eta": 2.3012, "rho": -0.4176, "v0": 0.1838, "alpha": 1.0},
    {"id": "exp_18_Feller_fail_alpha=0.9", "kappa": 6.5482, "theta": 0.0731, "eta": 2.3012, "rho": -0.4176, "v0": 0.1838, "alpha": 0.9},
    {"id": "exp_19_Feller_fail_alpha=0.8", "kappa": 6.5482, "theta": 0.0731, "eta": 2.3012, "rho": -0.4176, "v0": 0.1838, "alpha": 0.8},
    {"id": "exp_20_Feller_fail_alpha=0.7", "kappa": 6.5482, "theta": 0.0731, "eta": 2.3012, "rho": -0.4176, "v0": 0.1838, "alpha": 0.7},
    {"id": "exp_21_Feller_fail_alpha=0.6", "kappa": 6.5482, "theta": 0.0731, "eta": 2.3012, "rho": -0.4176, "v0": 0.1838, "alpha": 0.6},
    {"id": "exp_22_Feller_fail_alpha=0.5", "kappa": 6.5482, "theta": 0.0731, "eta": 2.3012, "rho": -0.4176, "v0": 0.1838, "alpha": 0.5},
    {"id": "exp_23_Feller_fail_alpha=0.4", "kappa": 6.5482, "theta": 0.0731, "eta": 2.3012, "rho": -0.4176, "v0": 0.1838, "alpha": 0.4},
    {"id": "exp_24_Feller_fail_alpha=0.3", "kappa": 6.5482, "theta": 0.0731, "eta": 2.3012, "rho": -0.4176, "v0": 0.1838, "alpha": 0.3}
]

print("=== GENERATION VOLATILITY SMILES + RSE + MC-MOMENTS (.npz) ===")
total_start = time.time()

for idx, exp in enumerate(experiments, 1):
    exp_id = exp["id"]
    kappa_val = exp["kappa"]
    theta_val = exp["theta"]
    eta_val = exp["eta"]
    rho_val = exp["rho"]
    v0_val = exp["v0"]
    alpha_val = exp["alpha"]

    print(f"\n[{idx}/24] Generating {exp_id}")
    t0 = time.time()

    all_smiles, all_rse, all_k_grids, all_moments, all_moments_se = volatility_smile.build_volatility_smile(
        S0, v0_val, kappa_val, theta_val, eta_val, rho_val, r, T_array, dt, ds, alpha_val, n_paths
    )
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s. Generated {all_smiles.shape[0]} smiles & RSE for T={T_array}.")

    filepath = os.path.join(output_dir, f"{exp_id}.npz")
    np.savez(
        filepath,
        T_array=np.array(T_array),
        k_grids=all_k_grids, 
        all_smiles=all_smiles,
        all_rse=all_rse,
        all_moments = all_moments,
        all_moments_se = all_moments_se,
        S0=S0,
        v0=v0_val,
        kappa=kappa_val,
        theta=theta_val,
        eta=eta_val,
        rho=rho_val,
        r=r,
        dt=dt,
        ds=ds,
        alpha=alpha_val,
        n_paths=n_paths
    )

total_elapsed = time.time() - total_start
print(f"\n==========================================")
print(f"ALL 24 NEW DATASETS SUCCESSFULLY SAVED IN .npz FORMAT!")
print(f"Total time elapsed: {total_elapsed / 60:.2f} minutes.")
print(f"Dataset folder: {os.path.abspath(output_dir)}")
