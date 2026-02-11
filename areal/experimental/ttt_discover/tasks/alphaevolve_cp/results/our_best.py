import numpy as np
from scipy.optimize import minimize
 

def run_packing():
    def initial_centers():
        n = 26
        rows = [5,5,6,5,5]
        r_initial = 0.1
        centers = []
        for row_idx, cols_in_row in enumerate(rows):
            y = r_initial + 2 * r_initial * row_idx
            if row_idx % 2 == 0:
                x_start = r_initial
            else:
                x_start = r_initial + r_initial
            for col in range(cols_in_row):
                x = x_start + 2 * r_initial * col
                centers.append([x, y])
        return np.array(centers)
    
    centers_initial = initial_centers()
    radii_initial = np.full(26, 0.1)
    variables_initial = np.concatenate([centers_initial.flatten(), radii_initial])

    def objective(vars):
        n = 26
        centers = vars[:2*n].reshape((n, 2))
        radii = vars[2*n:]
        return -np.sum(radii)

    constraints = []

    for i in range(26):
        def c1(vars, i=i): return vars[2*i] - vars[2*26 + i]
        def c2(vars, i=i): return (1 - vars[2*26 + i]) - vars[2*i]
        def c3(vars, i=i): return vars[2*i + 1] - vars[2*26 + i]
        def c4(vars, i=i): return (1 - vars[2*26 + i]) - vars[2*i + 1]
        constraints.append({'type': 'ineq', 'fun': c1})
        constraints.append({'type': 'ineq', 'fun': c2})
        constraints.append({'type': 'ineq', 'fun': c3})
        constraints.append({'type': 'ineq', 'fun': c4})

    for i in range(26):
        for j in range(i + 1, 26):
            def dist_con(vars, i=i, j=j):
                xi = vars[2*i]
                yi = vars[2*i + 1]
                xj = vars[2*j]
                yj = vars[2*j + 1]
                ri = vars[2*26 + i]
                rj = vars[2*26 + j]
                return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
            constraints.append({'type': 'ineq', 'fun': dist_con})

    bounds = []
    for _ in range(2*26):
        bounds.append((0.0, 1.0))  # x, y
    for _ in range(26):
        bounds.append((0.0, None))  # radii

    result = minimize(
        objective,
        variables_initial,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-6}
    )

    if not result.success:
        print("result not successful")
        return (centers_initial, radii_initial, 2.6)

    final_vars = result.x
    n = 26
    final_centers = final_vars[:2*n].reshape((n, 2))
    final_radii = final_vars[2*n:]
    sum_radii = np.sum(final_radii)
    return (final_centers, final_radii, sum_radii)