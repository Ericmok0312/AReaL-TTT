import os
import sys
import json

from circle_packing_evaluator import evaluate, run_with_timeout
from circle_packing_evaluator import validate_packing_ae


def main():
    program_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(os.path.dirname(__file__), "new_our_best.py")
    )
    out_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.join(os.path.dirname(__file__), "our_best_circles.json")
    )
    
    # Read and print the program
    with open(program_path, "r") as f:
        print(f.read())

    metrics = evaluate(program_path)
    print(json.dumps(metrics, indent=2, sort_keys=True))

    # Also save the constructed circles (x, y, r) and sum of radii
    try:
        centers, radii, sum_r = run_with_timeout(program_path, timeout_seconds=600)
        circles = [[float(x), float(y), float(r)] for (x, y), r in zip(centers, radii)]
        print("AE validation: ", validate_packing_ae(centers, radii))
        
        with open(out_path, "w") as f:
            json.dump({"circles": circles, "sum_radii": float(sum_r)}, f, indent=2, sort_keys=True)
        print(f"saved: {out_path}")
    except Exception as e:
        print(f"save_failed: {e}")


if __name__ == "__main__":
    main()


