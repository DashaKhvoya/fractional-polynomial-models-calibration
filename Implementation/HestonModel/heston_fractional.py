import numpy as np
import subordinator
import heston_classical

def single_path_fractional_heston(S0, v0, kappa, theta, eta, rho, r, T, dt, ds, alpha):
    if alpha == 1.0:
        # Standard Heston model (no subordinator needed)
        t_grid, S_path, V_path = heston_classical.single_path_heston(S0, v0, kappa, theta, eta, rho, r, T, dt)
        L_t = t_grid.copy()
            
        return t_grid, L_t, S_path, V_path

    # 1. Obtain the physical time grid, the auxiliary grid and the trajectory L_t
    t_grid, s_grid, L_t = subordinator.inverse_subordinator(alpha, T, dt, ds)
    
    S_aux = np.zeros_like(s_grid)
    V_aux = np.zeros_like(s_grid)
    S_aux[0] = S0
    V_aux[0] = v0

    N_steps = len(s_grid)-1

    dB1_array = np.random.normal(0.0, np.sqrt(ds), N_steps)
    dB2_array = np.random.normal(0.0, np.sqrt(ds), N_steps)
    dW_array = rho * dB1_array + np.sqrt(1.0 - rho**2) * dB2_array # corr(dB1,dW) = \rho

    # 2. Simulate classical Heston in auxiliary time s
    for i in range(N_steps):
        dB1 = dB1_array[i]
        dW = dW_array[i]

        v_pos = max(V_aux[i], 0.0)
        V_aux[i+1] = V_aux[i] + (kappa * (theta - v_pos) * ds + eta * np.sqrt(v_pos) * dW)
        S_aux[i+1] = np.exp(np.log(S_aux[i]) - 0.5 * v_pos * ds + np.sqrt(v_pos) * dB1)

    # 3. Time fix (s-time to t-time)
    S_path = np.zeros_like(t_grid)
    V_path = np.zeros_like(t_grid)

    for i in range(len(t_grid)):
        S_path[i] = S_aux[int(round(L_t[i] / ds))] * np.exp(r * t_grid[i])
        V_path[i] = V_aux[int(round(L_t[i] / ds))]

    return t_grid, L_t, S_path, V_path

