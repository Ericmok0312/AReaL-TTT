
import numpy as np
from scipy.optimize import minimize


def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True


def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    import math
    n = 32
    rows = 6
    cols = 6
    centers_initial = []
    for i in range(rows):
        for j in range(cols):
            x = j / (cols - 1) if cols > 1 else 0.5
            y = i / (rows - 1) if rows > 1 else 0.5
            if i % 2 == 1:
                x += 0.5 / (cols - 1) if cols > 1 else 0
            if len(centers_initial) < n:
                centers_initial.append([x, y])
            else:
                break
        if len(centers_initial) >= n:
            break
    centers_initial = np.array(centers_initial)
    
    radii_initial = []
    for k in range(n):
        x, y = centers_initial[k]
        dist_edge = min(x, 1 - x, y, 1 - y)
        dist_other = []
        for l in range(n):
            if k == l:
                continue
            dx = x - centers_initial[l, 0]
            dy = y - centers_initial[l, 1]
            dist = np.sqrt(dx**2 + dy**2)
            dist_other.append(dist)
        dist_to_other = np.min(dist_other) if dist_other else float('inf')
        r_initial = min(dist_edge, dist_to_other) / 2
        radii_initial.append(r_initial)
    radii_initial = np.array(radii_initial)
    
    x0 = np.concatenate([centers_initial.flatten(), radii_initial])

    def objective(x):
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        return -np.sum(radii)
    
    def constraint(x):
        centers = x[:2*n].reshape(n, 2)
        radii = x[2*n:]
        constraints = []
        for i in range(n):
            x_i, y_i = centers[i]
            r_i = radii[i]
            constraints.append(x_i - r_i)
            constraints.append(1 - x_i - r_i)
            constraints.append(y_i - r_i)
            constraints.append(1 - y_i - r_i)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                dist = np.sqrt(dx**2 + dy**2)
                constraints.append(dist - (radii[i] + radii[j]))
        return np.array(constraints)

    bounds = []
    for i in range(2*n):
        bounds.append((0, 1))
    for i in range(n):
        bounds.append((0, None))
    
    res = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': constraint}
    )
    
    if not res.success:
        print("Optimization failed:", res.message)
        return None

    centers_opt = res.x[:2*n].reshape(n, 2)
    radii_opt = res.x[2*n:]
    sum_radii = np.sum(radii_opt)

    if validate_packing(centers_opt, radii_opt):
        return (centers_opt, radii_opt, sum_radii)
    else:
        print("Validation failed after optimization")
        return None
