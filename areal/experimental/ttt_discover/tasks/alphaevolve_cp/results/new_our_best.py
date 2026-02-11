import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # Generate initial guess with a grid-based layout
    rows = [5, 5, 5, 6, 5]
    y_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    initial_radii = [0.05] * 26
    initial_centers = []
    for row_idx, row_length in enumerate(rows):
        y = y_positions[row_idx]
        r = 0.05
        width_available = 1.0 - 2 * r
        spacing = width_available / (row_length - 1) if row_length > 1 else 0.0
        for col_idx in range(row_length):
            x = r + col_idx * spacing
            initial_centers.append([x, y])
    
    # Create the initial variables array
    initial_vars = []
    for center in initial_centers:
        initial_vars.append(center[0])
        initial_vars.append(center[1])
    initial_vars.extend(initial_radii)
    
    # Objective function to maximize sum of radii
    def objective_function(vars):
        return -np.sum(vars[52:52 + 26])  # Negative for minimization
    
    # Generate constraints
    constraints = []
    
    # Constraints for centers within square and radii non-negativity
    for i in range(26):
        def x_lower_bound(vars, i=i):
            x = vars[2*i]
            r = vars[52 + i]
            return x - r
        constraints.append({'type': 'ineq', 'fun': x_lower_bound})
        
        def x_upper_bound(vars, i=i):
            x = vars[2*i]
            r = vars[52 + i]
            return 1.0 - x - r
        constraints.append({'type': 'ineq', 'fun': x_upper_bound})
        
        def y_lower_bound(vars, i=i):
            y = vars[2*i + 1]
            r = vars[52 + i]
            return y - r
        constraints.append({'type': 'ineq', 'fun': y_lower_bound})
        
        def y_upper_bound(vars, i=i):
            y = vars[2*i + 1]
            r = vars[52 + i]
            return 1.0 - y - r
        constraints.append({'type': 'ineq', 'fun': y_upper_bound})
    
    # Pairwise distance constraints
    for i in range(26):
        for j in range(i + 1, 26):
            def dist_constraint(vars, i=i, j=j):
                xi = vars[2*i]
                yi = vars[2*i + 1]
                xj = vars[2*j]
                yj = vars[2*j + 1]
                ri = vars[52 + i]
                rj = vars[52 + j]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx**2 + dy**2)
                return dist - (ri + rj)
            constraints.append({'type': 'ineq', 'fun': dist_constraint})
    
    # Variable bounds
    bounds = []
    for _ in range(2 * 26):
        bounds.append((0.0, 1.0))  # Center coordinates
    for _ in range(26):
        bounds.append((0.0, None))  # Radii
    
    # Perform optimization
    result = minimize(
        objective_function,
        initial_vars,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1500, 'ftol': 1e-12}
    )
    
    if not result.success:
        # Use initial guess if optimization fails
        centers = np.array(initial_centers)
        radii = np.array(initial_radii)
        sum_radii = sum(radii)
    else:
        optimized_vars = result.x
        centers = np.array([[optimized_vars[2*i], optimized_vars[2*i+1]] for i in range(26)])
        radii = np.array(optimized_vars[52:52+26])
        sum_radii = np.sum(radii)
    
    return (centers, radii, sum_radii)