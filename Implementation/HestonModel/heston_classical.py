import numpy as np

def single_path_heston(S0, v0, kappa, theta, eta, rho, r, T, dt):
    t_grid = np.arange(0, T + dt/2, dt) # add dt/2 for T to be included in array
    N_steps = len(t_grid) - 1
    
    S_path = np.zeros_like(t_grid)
    V_path = np.zeros_like(t_grid)
    V_path[0] = v0
    S_path[0] = S0

    dB1_array = np.random.normal(0.0, np.sqrt(dt), N_steps)
    dB2_array = np.random.normal(0.0, np.sqrt(dt), N_steps)
    dW_array = rho * dB1_array + np.sqrt(1.0 - rho**2) * dB2_array # corr(dB1,dW) = \rho
    
    for i in range(N_steps):
        dB1 = dB1_array[i]
        dW = dW_array[i]

        v_pos = max(V_path[i], 0.0)
        
        V_path[i+1] = V_path[i] + (kappa * (theta - v_pos) * dt + eta * np.sqrt(v_pos) * dW)
        S_path[i+1] = np.exp(np.log(S_path[i]) + (r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dB1)
        
    return t_grid, S_path, V_path