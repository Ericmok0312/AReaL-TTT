#!/usr/bin/env python3
"""
TTT-Discover Training Logger

用于记录训练过程中的 reward 分布数据，支持 checkpoint 恢复。
"""

import json
import os
import pickle
from typing import Any, Optional

import numpy as np
import torch.distributed as dist

from areal.utils import logging

logger = logging.getLogger("ttt_logger")


class TTTTrainingLogger:
    """
    TTT-Discover 训练过程记录器
    
    记录每个 step 的 reward 分布数据，支持：
    - 按指定 steps 记录 snapshot
    - 保存/加载 checkpoint
    - 分布式训练下的数据聚合
    
    Example:
        >>> history_logger = TTTTrainingLogger(
        ...     save_steps=[0, 9, 24, 49],
        ...     output_dir="./outputs",
        ...     is_dp_head=True,
        ... )
        >>> 
        >>> for step in range(50):
        ...     rollouts = generate_rollouts(...)
        ...     rewards = [r for _, r in rollouts]
        ...     
        ...     # 记录当前 step 数据
        ...     history_logger.record_step(
        ...         step=step,
        ...         rewards=rewards,
        ...         best_solution=find_best_solution(rollouts),
        ...     )
        ...     
        ...     # 训练代码...
        >>> 
        >>> # 保存最终历史
        >>> history_logger.save()
    """
    
    def __init__(
        self,
        save_steps: list[int],
        output_dir: str,
        is_dp_head: bool = True,
        filename: str = "training_history.pkl",
        checkpoint_filename: str = "training_history_checkpoint.pkl",
        aggregate_distributed: bool = True,
    ):
        """
        Args:
            save_steps: 需要保存 snapshot 的 step 列表，如 [0, 9, 24, 49]
            output_dir: 输出目录
            is_dp_head: 是否为 DP head rank（只有 head 会保存文件）
            filename: 最终保存的文件名
            checkpoint_filename: checkpoint 文件名
            aggregate_distributed: 是否在分布式环境下聚合所有 ranks 的数据
        """
        self.save_steps = sorted(set(save_steps))
        self.output_dir = output_dir
        self.is_dp_head = is_dp_head
        self.filename = filename
        self.checkpoint_filename = checkpoint_filename
        self.aggregate_distributed = aggregate_distributed
        
        # 训练历史数据（从 checkpoint 恢复时会追加）
        self.history: dict[str, Any] = {}
        self.best_of_n_data: Optional[dict[str, Any]] = None
        
        # 当前最佳解（跨所有 steps）
        self.overall_best_reward: float = float('-inf')
        self.overall_best_solution: Optional[dict] = None
        self.overall_best_step: int = -1
        
        # 尝试从 checkpoint 恢复（支持断点续训追加记录）
        if self.is_dp_head:
            loaded = self._try_load_checkpoint()
            if loaded:
                logger.info(f"[TTTLogger] Resumed from checkpoint. "
                           f"Existing history has {len(self.history)} snapshots, "
                           f"overall_best={self.overall_best_reward:.4f} at step {self.overall_best_step}")
            else:
                logger.info(f"[TTTLogger] Starting fresh training history")
    
    def _try_load_checkpoint(self) -> bool:
        """尝试从 checkpoint 恢复历史记录"""
        checkpoint_path = os.path.join(self.output_dir, self.checkpoint_filename)
        if not os.path.exists(checkpoint_path):
            logger.info(f"[TTTLogger] No checkpoint found at {checkpoint_path}")
            return False
        
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            
            self.history = data.get('history', {})
            self.best_of_n_data = data.get('best_of_n', None)
            self.overall_best_reward = data.get('overall_best_reward', float('-inf'))
            self.overall_best_solution = data.get('overall_best_solution', None)
            self.overall_best_step = data.get('overall_best_step', -1)
            
            logger.info(
                f"[TTTLogger] Loaded checkpoint from {checkpoint_path}\n"
                f"  - Restored {len(self.history)} step snapshots\n"
                f"  - Overall best reward: {self.overall_best_reward:.4f} (step {self.overall_best_step})"
            )
            return True
            
        except Exception as e:
            logger.warning(f"[TTTLogger] Failed to load checkpoint: {e}")
            return False
    
    def _aggregate_rewards(self, local_rewards: list[float]) -> list[float]:
        """
        在分布式环境下聚合所有 ranks 的 rewards
        
        Args:
            local_rewards: 当前 rank 的 rewards 列表
            
        Returns:
            聚合后的所有 ranks 的 rewards 列表（仅在 DP head 上有效）
        """
        if not self.aggregate_distributed or not dist.is_initialized():
            return local_rewards
        
        world_size = dist.get_world_size()
        if world_size <= 1:
            return local_rewards
        
        # 将所有 ranks 的 rewards 转换为 tensor 并聚合
        # 使用 all_gather_object 来收集变长列表
        all_rewards = [None] * world_size
        dist.all_gather_object(all_rewards, local_rewards)
        
        # 展平为单个列表
        aggregated = []
        for rewards in all_rewards:
            if rewards is not None:
                aggregated.extend(rewards)
        
        return aggregated
    
    def record_step(
        self,
        step: int,
        rewards: list[float],
        best_solution: Optional[dict] = None,
        additional_metrics: Optional[dict] = None,
        rollout_metadata: Optional[list[dict]] = None,
        puct_analysis_data: Optional[dict] = None,
    ) -> bool:
        """
        记录单个 step 的数据
        
        Args:
            step: 当前 step 编号
            rewards: 当前 step 的所有 rollout rewards（当前 rank）
            best_solution: 当前 step 的最佳解（可选字典，包含 code, value, observation, construction 等）
            additional_metrics: 额外指标（可选）
            rollout_metadata: 每个 rollout 的详细元数据（parent_id, exec_time_ms, staleness 等）
            puct_analysis_data: PUCT 行为分析数据（用于计算三个核心指标）
            
        Returns:
            是否成功记录了 snapshot（只有指定的 save_steps 才会记录）
        """
        # 聚合分布式数据
        aggregated_rewards = self._aggregate_rewards(rewards)
        
        # 计算统计信息
        rewards_array = np.array(aggregated_rewards)
        max_reward = float(rewards_array.max())
        mean_reward = float(rewards_array.mean())
        min_reward = float(rewards_array.min())
        std_reward = float(rewards_array.std())
        
        # 更新全局最佳
        if max_reward > self.overall_best_reward:
            self.overall_best_reward = max_reward
            self.overall_best_solution = best_solution
            self.overall_best_step = step
        
        # 只在指定的 steps 保存 snapshot
        if step not in self.save_steps:
            return False
        
        # 构建 snapshot 数据
        key = f"step_{step}"
        snapshot = {
            "step": step,
            "rewards": aggregated_rewards,
            "max_reward": max_reward,
            "mean_reward": mean_reward,
            "min_reward": min_reward,
            "std_reward": std_reward,
            "num_rollouts": len(aggregated_rewards),
            "best_solution": best_solution,
        }
        
        # 添加额外指标
        if additional_metrics:
            snapshot["metrics"] = additional_metrics
        
        # 添加 rollout 元数据（用于分析 parent complexity vs execution time）
        if rollout_metadata:
            snapshot["rollout_metadata"] = rollout_metadata
        
        # 添加 PUCT 分析数据（用于后续计算三个核心指标）
        if puct_analysis_data:
            snapshot["puct_analysis"] = puct_analysis_data
        
        self.history[key] = snapshot
        
        if self.is_dp_head:
            logger.info(
                f"[TTTLogger] Recorded snapshot for step {step}\n"
                f"  - Rollouts: {len(aggregated_rewards)}\n"
                f"  - Reward: mean={mean_reward:.4f}, max={max_reward:.4f}, std={std_reward:.4f}"
            )
            if rollout_metadata:
                n_with_meta = len(rollout_metadata)
                avg_staleness = sum(m['staleness'] for m in rollout_metadata) / n_with_meta if n_with_meta > 0 else 0
                logger.info(f"  - Metadata: {n_with_meta} rollouts with staleness avg={avg_staleness:.2f}")
        
        return True
    
    def record_best_of_n(
        self,
        rewards: list[float],
        best_solution: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        记录 Best-of-N 基线数据
        
        Args:
            rewards: Best-of-N 采样的所有 rewards
            best_solution: Best-of-N 的最佳解（字典，包含 code, value, observation, construction 等）
            metadata: 额外元数据（如 n_samples, model_name 等）
        """
        # 聚合分布式数据
        aggregated_rewards = self._aggregate_rewards(rewards)
        
        rewards_array = np.array(aggregated_rewards)
        self.best_of_n_data = {
            "rewards": aggregated_rewards,
            "max_reward": float(rewards_array.max()),
            "mean_reward": float(rewards_array.mean()),
            "min_reward": float(rewards_array.min()),
            "std_reward": float(rewards_array.std()),
            "num_samples": len(aggregated_rewards),
            "best_solution": best_solution,
            "metadata": metadata or {},
        }
        
        if self.is_dp_head:
            logger.info(
                f"[TTTLogger] Recorded Best-of-N baseline\n"
                f"  - Samples: {len(aggregated_rewards)}\n"
                f"  - Reward: mean={self.best_of_n_data['mean_reward']:.4f}, "
                f"max={self.best_of_n_data['max_reward']:.4f}"
            )
    
    def save_checkpoint(self) -> Optional[str]:
        """
        保存 checkpoint（用于训练中断恢复）
        
        Returns:
            保存的文件路径（如果不是 DP head 则返回 None）
        """
        if not self.is_dp_head:
            return None
        
        os.makedirs(self.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.output_dir, self.checkpoint_filename)
        
        data = {
            'history': self.history,
            'best_of_n': self.best_of_n_data,
            'overall_best_reward': self.overall_best_reward,
            'overall_best_solution': self.overall_best_solution,
            'overall_best_step': self.overall_best_step,
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        
        return checkpoint_path
    
    def save(self, also_save_json: bool = True) -> Optional[str]:
        """
        保存最终训练历史
        
        Args:
            also_save_json: 是否同时保存 JSON 格式（便于查看）
            
        Returns:
            保存的文件路径（如果不是 DP head 则返回 None）
        """
        if not self.is_dp_head:
            return None
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 构建最终数据
        data = {
            'history': self.history,
            'metadata': {
                'save_steps': self.save_steps,
                'num_snapshots': len(self.history),
                'overall_best_reward': self.overall_best_reward,
                'overall_best_step': self.overall_best_step,
            }
        }
        
        if self.best_of_n_data:
            data['best_of_n'] = self.best_of_n_data
        
        if self.overall_best_solution:
            data['metadata']['overall_best_solution'] = self.overall_best_solution
        
        # 保存为 pickle（完整数据，包含所有 rewards 列表）
        pkl_path = os.path.join(self.output_dir, self.filename)
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(
            f"[TTTLogger] Saved training history to {pkl_path}\n"
            f"  - Snapshots: {len(self.history)}\n"
            f"  - Overall best: {self.overall_best_reward:.4f} at step {self.overall_best_step}"
        )
        
        # 同时保存为 JSON（不包含完整的 rewards 列表，避免文件过大）
        if also_save_json:
            json_data = {
                'metadata': data['metadata'],
                'snapshots_summary': {
                    key: {
                        'step': snap['step'],
                        'num_rollouts': snap['num_rollouts'],
                        'max_reward': snap['max_reward'],
                        'mean_reward': snap['mean_reward'],
                        'std_reward': snap['std_reward'],
                        'metrics': snap.get('metrics', {}),
                    }
                    for key, snap in self.history.items()
                }
            }
            
            if self.best_of_n_data:
                json_data['best_of_n_summary'] = {
                    'num_samples': self.best_of_n_data['num_samples'],
                    'max_reward': self.best_of_n_data['max_reward'],
                    'mean_reward': self.best_of_n_data['mean_reward'],
                    'std_reward': self.best_of_n_data['std_reward'],
                }
            
            json_path = os.path.join(self.output_dir, self.filename.replace('.pkl', '.json'))
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            logger.info(f"[TTTLogger] Saved summary to {json_path}")
        
        return pkl_path
    
    def get_summary(self) -> dict:
        """获取训练历史摘要"""
        return {
            'num_snapshots': len(self.history),
            'save_steps': self.save_steps,
            'recorded_steps': sorted([snap['step'] for snap in self.history.values()]),
            'overall_best_reward': self.overall_best_reward,
            'overall_best_step': self.overall_best_step,
            'has_best_of_n': self.best_of_n_data is not None,
        }


def load_training_history(path: str) -> dict:
    """
    加载训练历史文件
    
    Args:
        path: pkl 文件路径
        
    Returns:
        训练历史字典
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
