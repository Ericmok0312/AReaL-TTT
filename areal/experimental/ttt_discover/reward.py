"""
Standard AReaL-style reward functions for TTT-Discover.

These functions follow AReaL's reward API convention:
    reward_fn(prompt, completions, prompt_ids, completion_ids, **data) -> float

Usage:
    from areal.experimental.ttt_discover.reward import tttd_reward_fn
    
    # In workflow
    reward = await async_reward_fn(
        prompt_str,
        completion_str,
        prompt_ids,
        completion_ids,
        _env=env,      # Extra data for TTT-Discover
        _state=state,
    )
"""

from areal.utils import logging

logger = logging.getLogger("TTTDReward")


def tttd_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    **data,
) -> tuple[float, Any, str]:
    """
    TTT-Discover reward function returning full execution result.
    
    This is an extended version of AReaL's standard reward function that returns
    additional metadata to avoid re-executing code in the workflow.
    
    Args:
        prompt: The prompt string (decoded from prompt_ids).
        completions: The generated completion string (decoded from completion_ids).
        prompt_ids: List of token IDs for the prompt.
        completion_ids: List of token IDs for the completion.
        **data: Additional data passed from the workflow. Required fields:
            - _env: BaseEnv instance for code extraction and execution.
            - _state: State instance for the current task.
    
    Returns:
        tuple: (reward, EnvResult, code)
            - reward: float, the computed reward (higher is better)
            - result: EnvResult, full execution result with metadata
            - code: str, extracted code string
            
    Example:
        >>> reward, result, code = tttd_reward_fn(
        ...     prompt="Solve AC1...",
        ...     completions="```python\ndef propose_candidate(): ...",
        ...     prompt_ids=[1, 2, 3],
        ...     completion_ids=[4, 5, 6],
        ...     _env=inequalities_env,
        ...     _state=current_state,
        ... )
    """
    env = data.get("_env")
    state = data.get("_state")
    
    if env is None:
        logger.error("Missing '_env' in data. Pass env via _env=env.")
        result = env.get_failure_result(state=None, fail_type="missing_env") if env else None
        return 0.0, result, ""
    
    if state is None:
        logger.error("Missing '_state' in data. Pass state via _state=state.")
        result = env.get_failure_result(state=None, fail_type="missing_state")
        return 0.0, result, ""
    
    try:
        # Extract code from completion
        code = env.extract_code(completions)
        
        if code is None:
            logger.warning("Code extraction failed from completion")
            result = env.get_failure_result(
                state=state,
                fail_type="code_extraction_failed",
            )
            return 0.0, result, ""
        
        # Execute code in environment
        result = env.execute(code, state)
        
        # Return full tuple to avoid re-execution in workflow
        return float(result.reward), result, code
        
    except Exception as e:
        logger.warning(f"Exception in tttd_reward_fn: {e}", exc_info=True)
        result = env.get_failure_result(
            state=state,
            fail_type="execution_error",
            error_msg=str(e),
        )
        return 0.0, result, ""


def tttd_reward_fn_with_validation(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    **data,
) -> float:
    """
    Extended reward function with detailed validation and logging.
    
    Same as tttd_reward_fn but with additional validation checks and debug logging.
    Useful for debugging new environments.
    
    Args:
        Same as tttd_reward_fn.
        
    Returns:
        float: The computed reward.
    """
    env = data.get("_env")
    state = data.get("_state")
    
    logger.debug(f"tttd_reward_fn_with_validation called")
    logger.debug(f"  prompt length: {len(prompt)}")
    logger.debug(f"  completion length: {len(completions)}")
    logger.debug(f"  env type: {type(env).__name__ if env else 'None'}")
    logger.debug(f"  state type: {type(state).__name__ if state else 'None'}")
    
    if env is None or state is None:
        logger.error(f"Missing required data: env={env is not None}, state={state is not None}")
        return 0.0
    
    try:
        code = env.extract_code(completions)
        
        if code is None:
            logger.warning("Code extraction returned None")
            return 0.0
        
        logger.debug(f"Extracted code length: {len(code)}")
        
        result = env.execute(code, state)
        
        logger.debug(f"Execution result: reward={result.reward}, valid={result.is_valid}, fail_type={result.fail_type}")
        
        return float(result.reward)
        
    except Exception as e:
        logger.error(f"Error in reward computation: {e}", exc_info=True)
        return 0.0
