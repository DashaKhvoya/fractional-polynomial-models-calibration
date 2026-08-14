import numpy as np

def sample_stable_cms(alpha, scale, size):
    V = np.random.uniform(-np.pi / 2.0, np.pi / 2.0, size=size)
    W = np.random.exponential(1.0, size=size)
    
    inv_alpha = 1.0 / alpha
    val0 = np.pi / 2.0
    
    num = np.sin(alpha * (V + val0))
    den = (np.cos(V)) ** inv_alpha
    term2 = (np.cos(V - alpha * (V + val0)) / W) ** ((1.0 - alpha) / alpha)
    
    return scale * (num / den) * term2


def inverse_subordinator(alpha, T, dt, ds):
    scale_factor = ds ** (1.0 / alpha)
    
    n_batch = int(T / ds * 1.5) + 200
    sigma_list = [0.0]
    curr_sigma = 0.0
    
    while curr_sigma <= T:
        batch = sample_stable_cms(alpha, scale_factor, n_batch)
        cumsum_batch = np.cumsum(batch) + curr_sigma
        sigma_list.extend(cumsum_batch.tolist())
        curr_sigma = sigma_list[-1]
        
    sigma_s = np.array(sigma_list)
    s_grid = np.arange(len(sigma_s)) * ds
    
    t_grid = np.arange(0, T + dt / 2.0, dt)
    
    idx = np.searchsorted(sigma_s, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(s_grid) - 1)
    
    L_t = s_grid[idx]
    
    return t_grid, s_grid, L_t