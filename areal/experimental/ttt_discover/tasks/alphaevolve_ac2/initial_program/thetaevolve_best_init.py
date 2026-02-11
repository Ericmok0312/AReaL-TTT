"""Gradient ascent for C₂ autoconvolution lower-bound optimization."""
import numpy as np
from tasks.alphaevolve_ac2.ae_seq import height_sequence_2


def evaluate_sequence(sequence: list[float]) -> float:
    # Verify that the input is a list
    if not isinstance(sequence, list):
        return np.inf

    # Reject empty lists
    if not sequence:
        return np.inf

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types (as they are a subclass of int) and
        # any other non-integer/non-float types (like strings or complex numbers).
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return np.inf

        # Reject Not-a-Number (NaN) and infinity values.
        if np.isnan(x) or np.isinf(x):
            return np.inf

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    convolution_2 = np.convolve(sequence, sequence)
    # --- Security Checks ---

    # Calculate the 2-norm squared: ||f*f||_2^2
    num_points = len(convolution_2)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points) # Width of each interval
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):  # Iterate through intervals
        y1 = y_points[i]
        y2 = y_points[i+1]
        h = x_intervals[i]
        # Integral of (mx + c)^2 = h/3 * (y1^2 + y1*y2 + y2^2) where m = (y2-y1)/h, c = y1 - m*x1, interval is [x1, x2], y1 = mx1+c, y2=mx2+c
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    # Calculate the 1-norm: ||f*f||_1
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)

    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution_2))
    C_lower_bound = l2_norm_squared / (norm_1 * norm_inf)
    return C_lower_bound


def construct_function():
    """
    Principled local search for C2 autoconvolution optimization.
    
    Key insight: R = L2²/(L1·L∞). To maximize:
    - Concentrate mass at strategic positions (increases L2²/L1)
    - Avoid single dominant peak in convolution (reduces L∞)
    
    Strategy: Target high-value regions, use scaling/shifting perturbations.
    """
    import time
    rng = np.random.default_rng(42)
    
    best_h = np.array(height_sequence_2, dtype=np.float64)
    best_score = evaluate_sequence(best_h.tolist())
    N = len(best_h)
    
    # Find active region (non-zero support)
    active = np.where(best_h > 0.01)[0]
    a_min, a_max = active.min(), active.max()
    print(f"N={N} | Active region: [{a_min}, {a_max}] ({len(active)} pts) | Initial: {best_score:.6f}")
    
    t0 = time.time()
    improvements = 0
    
    def try_update(h_new):
        nonlocal best_h, best_score, improvements
        score = evaluate_sequence(np.clip(h_new, 0, None).tolist())
        if score > best_score:
            best_h, best_score = h_new.copy(), score
            improvements += 1
            return True
        return False
    
    peaks = np.where(best_h > np.percentile(best_h[active], 80))[0]
    
    for it in range(1, 31):
        # 1 eval: Scale peaks
        h_new = best_h.copy()
        h_new[peaks] *= rng.choice([0.995, 1.005])
        try_update(h_new)
        
        # 1 eval: Mass transfer
        if len(peaks) > 1:
            i, j = rng.choice(len(peaks), 2, replace=False)
            h_new = best_h.copy()
            h_new[peaks[i]] *= 0.98; h_new[peaks[j]] *= 1.02
            try_update(h_new)
        
        # 1 eval: Sharpen
        p = rng.choice(peaks)
        h_new = best_h.copy()
        h_new[max(0,p-10):min(N,p+10)] *= 0.99; h_new[p] *= 1.01
        try_update(h_new)
        
        # 1 eval: Local noise
        start = rng.integers(a_min, max(a_min+1, a_max-100))
        h_new = best_h.copy()
        h_new[start:start+100] += rng.normal(0, 0.003, 100)
        try_update(h_new)
        
        print(f"[{time.time()-t0:4.1f}s] {it}/30 | best: {best_score:.6f} | impr: {improvements}")
    
    print(f"Done in {time.time()-t0:.1f}s | Final: {best_score:.6f} | Δ improvements: {improvements}")
    return best_h.astype(np.float32)


if __name__ == "__main__":
    heights, r_value = construct_function()
    print(evaluate_sequence(heights.tolist()))
