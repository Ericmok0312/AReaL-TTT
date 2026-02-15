# Handle circular import gracefully - if case_runner is still being imported,
# we'll import it lazily when needed
try:
    from ale_bench.tool_wrappers.case_runner import run_cases
except (ImportError, AttributeError):
    # Module is still being imported, will be available later
    def run_cases(*args, **kwargs):
        from ale_bench.tool_wrappers.case_runner import run_cases
        return run_cases(*args, **kwargs)

try:
    from ale_bench.tool_wrappers.code_runner import run_code
except (ImportError, AttributeError):
    def run_code(*args, **kwargs):
        from ale_bench.tool_wrappers.code_runner import run_code
        return run_code(*args, **kwargs)

try:
    from ale_bench.tool_wrappers.input_generation import generate_inputs
except (ImportError, AttributeError):
    def generate_inputs(*args, **kwargs):
        from ale_bench.tool_wrappers.input_generation import generate_inputs
        return generate_inputs(*args, **kwargs)

try:
    from ale_bench.tool_wrappers.local_visualization import local_visualization
except (ImportError, AttributeError):
    def local_visualization(*args, **kwargs):
        from ale_bench.tool_wrappers.local_visualization import local_visualization
        return local_visualization(*args, **kwargs)

__all__ = ["generate_inputs", "local_visualization", "run_cases", "run_code"]
