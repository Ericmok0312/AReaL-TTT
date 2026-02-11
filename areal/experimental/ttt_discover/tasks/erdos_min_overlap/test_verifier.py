import numpy as np
from tasks.erdos_min_overlap.verifier import verify_c5_solution, evaluate_erdos_solution

def test_basic():
    n_points = 200
    h_values = np.ones(n_points) * 0.5
    dx = 2.0 / n_points
    
    integral = np.sum(h_values) * dx
    print(f"Integral: {integral}")
    assert np.isclose(integral, 1.0, atol=1e-3)
    
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    c5 = np.max(correlation)
    print(f"C5 bound: {c5}")
    
    result = verify_c5_solution(h_values, c5, n_points)
    print(f"Verification passed! C5 = {result}")
    
    combined = evaluate_erdos_solution(h_values, c5, n_points)
    print(f"Evaluated C5: {combined}")
    
    print("✓ Basic test passed!")

def test_invalid_integral():
    n_points = 200
    h_values = np.ones(n_points) * 0.8
    dx = 2.0 / n_points
    
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    c5 = np.max(correlation)
    
    try:
        verify_c5_solution(h_values, c5, n_points)
        print("ERROR: Should have raised ValueError for invalid integral")
        return False
    except ValueError as e:
        print(f"✓ Invalid integral test passed: {e}")
        return True

def test_out_of_bounds():
    n_points = 200
    h_values = np.ones(n_points) * 0.5
    h_values[0] = 1.5
    dx = 2.0 / n_points
    
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    c5 = np.max(correlation)
    
    try:
        verify_c5_solution(h_values, c5, n_points)
        print("ERROR: Should have raised ValueError for out of bounds")
        return False
    except ValueError as e:
        print(f"✓ Out of bounds test passed: {e}")
        return True

def test_nan():
    n_points = 200
    h_values = np.ones(n_points) * 0.5
    h_values[0] = np.nan
    dx = 2.0 / n_points
    
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    c5 = np.max(correlation)
    
    try:
        verify_c5_solution(h_values, c5, n_points)
        print("ERROR: Should have raised ValueError for NaN")
        return False
    except ValueError as e:
        print(f"✓ NaN test passed: {e}")
        return True

if __name__ == "__main__":
    tests = [
        test_basic,
        test_invalid_integral,
        test_out_of_bounds,
        test_nan,
    ]
    
    passed = sum(test() if test != test_basic else (test(), True)[1] for test in tests)
    total = len(tests)
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{total}")
    if passed == total:
        print("All tests passed! ✓")
    else:
        print(f"Some tests failed! {total - passed} failures")



