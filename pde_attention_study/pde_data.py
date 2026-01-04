"""Generate simple PDE data for attention study."""
import torch
import numpy as np

def generate_heat_equation_1d(n_samples=1000, n_points=64, n_timesteps=32):
    """
    Generate 1D heat equation solutions.
    u_t = alpha * u_xx
    
    Returns: (initial_conditions, solutions) 
        - initial_conditions: (n_samples, n_points)
        - solutions: (n_samples, n_timesteps, n_points)
    """
    x = np.linspace(0, 1, n_points)
    t = np.linspace(0, 0.1, n_timesteps)
    alpha = 0.01  # diffusion coefficient
    
    initial_conditions = []
    solutions = []
    
    for _ in range(n_samples):
        # Random Fourier initial condition
        n_modes = np.random.randint(1, 5)
        u0 = np.zeros(n_points)
        for k in range(1, n_modes + 1):
            amp = np.random.uniform(-1, 1)
            u0 += amp * np.sin(k * np.pi * x)
        
        # Analytical solution via Fourier
        solution = np.zeros((n_timesteps, n_points))
        for i, ti in enumerate(t):
            u = np.zeros(n_points)
            for k in range(1, n_modes + 1):
                amp = np.random.uniform(-1, 1) if i == 0 else 0
                # Fourier coefficient decays as exp(-alpha * k^2 * pi^2 * t)
                decay = np.exp(-alpha * (k * np.pi) ** 2 * ti)
                u += amp * decay * np.sin(k * np.pi * x)
            solution[i] = u if i == 0 else solution[0] * np.exp(-alpha * np.pi**2 * ti)
        
        # Simpler: just use finite differences for actual solution
        solution = solve_heat_fd(u0, alpha, n_timesteps, n_points)
        
        initial_conditions.append(u0)
        solutions.append(solution)
    
    return torch.tensor(np.array(initial_conditions), dtype=torch.float32), \
           torch.tensor(np.array(solutions), dtype=torch.float32)

def solve_heat_fd(u0, alpha, n_timesteps, n_points):
    """Solve heat equation using finite differences."""
    dx = 1.0 / (n_points - 1)
    dt = 0.1 / n_timesteps
    r = alpha * dt / dx**2
    
    u = u0.copy()
    solution = [u.copy()]
    
    for _ in range(n_timesteps - 1):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
        u_new[0] = 0  # boundary
        u_new[-1] = 0
        u = u_new
        solution.append(u.copy())
    
    return np.array(solution)

def generate_wave_equation_1d(n_samples=1000, n_points=64, n_timesteps=32):
    """Generate 1D wave equation data."""
    x = np.linspace(0, 1, n_points)
    c = 1.0  # wave speed
    
    initial_conditions = []
    solutions = []
    
    for _ in range(n_samples):
        # Gaussian bump initial condition
        x0 = np.random.uniform(0.3, 0.7)
        sigma = np.random.uniform(0.05, 0.15)
        u0 = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
        
        solution = solve_wave_fd(u0, c, n_timesteps, n_points)
        initial_conditions.append(u0)
        solutions.append(solution)
    
    return torch.tensor(np.array(initial_conditions), dtype=torch.float32), \
           torch.tensor(np.array(solutions), dtype=torch.float32)

def solve_wave_fd(u0, c, n_timesteps, n_points):
    """Solve wave equation using finite differences."""
    dx = 1.0 / (n_points - 1)
    dt = 0.1 / n_timesteps
    r = (c * dt / dx) ** 2
    
    u_prev = u0.copy()
    u = u0.copy()  # zero initial velocity
    solution = [u.copy()]
    
    for _ in range(n_timesteps - 1):
        u_new = np.zeros_like(u)
        u_new[1:-1] = 2*u[1:-1] - u_prev[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
        u_new[0] = 0
        u_new[-1] = 0
        u_prev = u.copy()
        u = u_new
        solution.append(u.copy())
    
    return np.array(solution)


def generate_ood_heat_data(ic_type='step', n_samples=100, n_points=64, n_timesteps=32):
    """Generate out-of-distribution initial conditions for testing generalization."""
    x = np.linspace(0, 1, n_points)
    alpha = 0.01
    
    initial_conditions = []
    solutions = []
    
    for _ in range(n_samples):
        if ic_type == 'step':
            # Step function
            pos = np.random.uniform(0.3, 0.7)
            u0 = (x > pos).astype(float) * np.random.uniform(0.5, 1.5)
            u0[0] = u0[-1] = 0
        elif ic_type == 'gaussian':
            # Gaussian bump (different from training sine waves)
            x0 = np.random.uniform(0.2, 0.8)
            sigma = np.random.uniform(0.05, 0.15)
            u0 = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
            u0[0] = u0[-1] = 0
        elif ic_type == 'triangle':
            # Triangle wave
            peak = np.random.uniform(0.3, 0.7)
            u0 = np.minimum(x / peak, (1 - x) / (1 - peak))
            u0 *= np.random.uniform(0.5, 1.5)
            u0[0] = u0[-1] = 0
        elif ic_type == 'multi_bump':
            # Multiple Gaussian bumps
            u0 = np.zeros(n_points)
            n_bumps = np.random.randint(2, 4)
            for _ in range(n_bumps):
                x0 = np.random.uniform(0.1, 0.9)
                sigma = np.random.uniform(0.03, 0.08)
                u0 += np.random.uniform(-1, 1) * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
            u0[0] = u0[-1] = 0
        else:
            raise ValueError(f"Unknown IC type: {ic_type}")
        
        solution = solve_heat_fd(u0, alpha, n_timesteps, n_points)
        initial_conditions.append(u0)
        solutions.append(solution)
    
    return torch.tensor(np.array(initial_conditions), dtype=torch.float32), \
           torch.tensor(np.array(solutions), dtype=torch.float32)


if __name__ == "__main__":
    ic, sol = generate_heat_equation_1d(n_samples=10)
    print(f"Initial conditions shape: {ic.shape}")
    print(f"Solutions shape: {sol.shape}")

