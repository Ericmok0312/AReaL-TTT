#!/usr/bin/env python3
"""
TTT-Discover Training Visualizer

用于绘制训练过程中的 reward 分布动态图（类似论文 Figure 1）
"""

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde

# 设置默认样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


class TTTVisualizer:
    """
    TTT-Discover 训练可视化器
    
    绘制训练过程中的 reward 分布 KDE 图，支持：
    - 多步骤分布叠加显示
    - Best-of-N 基线对比
    - Benchmark 阈值标注
    - 优化方向指示
    
    Example:
        >>> visualizer = TTTVisualizer(higher_is_better=True)
        >>> visualizer.plot_training_dynamics(
        ...     history_dict=training_history,
        ...     steps=[0, 9, 24, 49],
        ...     benchmark_value=2.635983,
        ...     output_path='training_dynamics.png',
        ... )
    """
    
    def __init__(self, higher_is_better: bool = True):
        """
        Args:
            higher_is_better: reward 是否越大越好
        """
        self.higher_is_better = higher_is_better
    
    def plot_training_dynamics(
        self,
        history_dict: dict,
        steps: list[int] = [0, 9, 24, 49],
        best_of_n_key: str = 'best_of_n',
        benchmark_value: Optional[float] = None,
        benchmark_label: str = 'Best Human',
        xlabel: str = 'Sum of Radii (higher is better →)',
        ylabel: str = 'Probability Density',
        title: str = 'TTT-Discover Training Dynamics',
        output_path: str = 'training_dynamics.png',
        figsize: tuple = (10, 6),
        dpi: int = 300,
        colors: Optional[list] = None,
        fill_alpha: float = 0.4,
        linewidth: float = 2.0,
        show_mean_lines: bool = False,
        show_arrow: bool = True,
        arrow_y_offset: float = 0.95,
        reward_transform: Optional[callable] = None,
        xlim: Optional[tuple[float, float]] = None,
    ) -> None:
        """
        绘制训练动态分布图（类似论文 Figure 1）
        
        Args:
            history_dict: 训练历史数据（TTTTrainingLogger 生成的格式）
            steps: 要绘制的 step 列表
            best_of_n_key: Best-of-N 数据在 history_dict 中的 key
            benchmark_value: Benchmark 阈值（如 Best Human 值）
            benchmark_label: Benchmark 标签
            xlabel: X 轴标签
            ylabel: Y 轴标签
            title: 图表标题
            output_path: 输出图片路径
            figsize: 图片尺寸
            dpi: 图片分辨率
            colors: 自定义颜色列表（默认使用 Blues_r colormap）
            fill_alpha: 填充区域透明度
            linewidth: 曲线线宽
            show_mean_lines: 是否显示均值垂直线
            show_arrow: 是否显示优化方向箭头
            arrow_y_offset: 箭头在 y 轴的位置（0-1 之间，相对于 y 轴范围）
            reward_transform: 可选的 reward 转换函数，如 lambda r: 1/r 用于 AC1
            xlim: 可选的 x 轴显示范围元组 (min, max)，用于聚焦有效值区域
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 准备颜色方案
        if colors is None:
            # 使用 Blues_r colormap，从深到浅
            cmap = sns.color_palette("Blues_r", n_colors=len(steps))
            colors = [cmap[i] for i in range(len(steps))]
        
        # 获取 history 数据
        history = history_dict.get('history', history_dict)
        
        # 跟踪 x 和 y 的范围用于设置坐标轴
        all_rewards = []
        max_density = 0.0
        
        # 绘制每个 step 的 KDE
        for idx, step in enumerate(steps):
            key = f"step_{step}"
            if key not in history:
                warnings.warn(f"Step {step} not found in history, skipping")
                continue
            
            step_data = history[key]
            rewards = np.array(step_data['rewards'])
            
            # 应用 reward transform（如 AC1: 1/reward）
            if reward_transform is not None:
                rewards = reward_transform(rewards)
            
            all_rewards.extend(rewards)
            
            # 计算 KDE
            kde, x_range, y_values = self._compute_kde(rewards)
            
            if kde is None:
                continue
            
            max_density = max(max_density, y_values.max())
            
            # 绘制 KDE 曲线和填充
            color = colors[idx]
            label = f"Step {step}"
            
            ax.plot(x_range, y_values, color=color, linewidth=linewidth, label=label)
            ax.fill_between(x_range, y_values, alpha=fill_alpha, color=color)
            
            # 可选：绘制均值线
            if show_mean_lines:
                raw_mean = step_data.get('mean_reward', np.mean(step_data['rewards']))
                if reward_transform is not None:
                    mean_val = reward_transform(raw_mean)
                else:
                    mean_val = raw_mean
                ax.axvline(
                    mean_val, 
                    color=color, 
                    linestyle='--', 
                    alpha=0.7,
                    linewidth=1.0,
                )
        
        # 绘制 Best-of-N 基线（灰色）
        if best_of_n_key in history_dict:
            bon_data = history_dict[best_of_n_key]
            if isinstance(bon_data, dict) and 'rewards' in bon_data:
                rewards = np.array(bon_data['rewards'])
                
                # 应用 reward transform
                if reward_transform is not None:
                    rewards = reward_transform(rewards)
                
                all_rewards.extend(rewards)
                
                kde, x_range, y_values = self._compute_kde(rewards)
                
                if kde is not None:
                    max_density = max(max_density, y_values.max())
                    ax.plot(
                        x_range, y_values, 
                        color='gray', 
                        linewidth=linewidth, 
                        linestyle='-',
                        label='Best-of-N (frozen policy)'
                    )
                    ax.fill_between(x_range, y_values, alpha=fill_alpha * 0.8, color='gray')
        
        # 绘制 Benchmark 参考线
        if benchmark_value is not None:
            ax.axvline(
                benchmark_value, 
                color='black', 
                linestyle='--', 
                linewidth=1.5,
                label=benchmark_label
            )
        
        # 添加优化方向箭头
        if show_arrow:
            self._add_optimization_arrow(ax, arrow_y_offset, max_density)
        
        # 设置坐标轴和标签
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 设置 y 轴从 0 开始
        ax.set_ylim(bottom=0)
        
        # 设置 x 轴显示范围（如果指定了 xlim）
        if xlim is not None:
            ax.set_xlim(xlim)
        
        # 添加图例
        ax.legend(loc='best', framealpha=0.9)
        
        # 移除顶部和右侧边框
        sns.despine()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[TTTVisualizer] Saved plot to {output_path}")
    
    def _compute_kde(
        self, 
        rewards: np.ndarray,
        num_points: int = 500,
        bw_adjust: float = 1.0,
    ) -> tuple:
        """
        计算 KDE
        
        Args:
            rewards: reward 数组
            num_points: KDE 曲线采样点数
            bw_adjust: 带宽调整因子
            
        Returns:
            (kde, x_range, y_values) 或 (None, None, None) 如果数据不足
        """
        if len(rewards) < 2:
            return None, None, None
        
        # 处理所有值相同的情况
        if np.std(rewards) < 1e-10:
            # 添加微小噪声
            rewards = rewards + np.random.normal(0, 1e-6, len(rewards))
        
        kde = gaussian_kde(rewards)
        kde.set_bandwidth(kde.factor * bw_adjust)
        
        # 确定 x 范围（稍微扩展边界）
        margin = (rewards.max() - rewards.min()) * 0.1
        x_min = rewards.min() - margin
        x_max = rewards.max() + margin
        x_range = np.linspace(x_min, x_max, num_points)
        
        y_values = kde(x_range)
        
        return kde, x_range, y_values
    
    def _add_optimization_arrow(
        self, 
        ax, 
        y_offset: float,
        max_density: float,
    ) -> None:
        """添加优化方向箭头"""
        # 获取当前 x 轴范围
        xlim = ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        
        # 箭头位置（右上角）
        if self.higher_is_better:
            # 箭头指向右侧（越大越好）
            arrow_start = xlim[0] + x_range * 0.75
            arrow_end = xlim[0] + x_range * 0.90
            direction = '→'
            text = 'better →'
        else:
            # 箭头指向左侧（越小越好）
            arrow_start = xlim[0] + x_range * 0.90
            arrow_end = xlim[0] + x_range * 0.75
            direction = '←'
            text = '← better'
        
        y_pos = max_density * y_offset
        
        # 绘制箭头
        ax.annotate(
            '',
            xy=(arrow_end, y_pos),
            xytext=(arrow_start, y_pos),
            arrowprops=dict(
                arrowstyle='->',
                color='darkred',
                lw=2.0,
            ),
        )
        
        # 添加文字
        text_x = (arrow_start + arrow_end) / 2
        ax.text(
            text_x, 
            y_pos + max_density * 0.05, 
            text,
            ha='center',
            va='bottom',
            fontsize=10,
            color='darkred',
            fontweight='bold',
        )
    
    def plot_reward_progression(
        self,
        history_dict: dict,
        output_path: str = 'reward_progression.png',
        figsize: tuple = (10, 6),
        dpi: int = 300,
        reward_transform: Optional[callable] = None,
    ) -> None:
        """
        绘制 reward 进展图（mean/max/std 随 step 变化）
        
        Args:
            history_dict: 训练历史数据
            output_path: 输出图片路径
            figsize: 图片尺寸
            dpi: 图片分辨率
            reward_transform: 可选的 reward 转换函数，如 lambda r: 1/r 用于 AC1
        """
        history = history_dict.get('history', history_dict)
        
        # 提取数据
        steps = []
        mean_rewards = []
        max_rewards = []
        std_rewards = []
        
        for key in sorted(history.keys()):
            if key.startswith('step_'):
                step_data = history[key]
                steps.append(step_data['step'])
                
                # 应用 reward transform
                if reward_transform is not None:
                    mean_rewards.append(reward_transform(step_data['mean_reward']))
                    max_rewards.append(reward_transform(step_data['max_reward']))
                    std_rewards.append(reward_transform(step_data.get('std_reward', 0)))
                else:
                    mean_rewards.append(step_data['mean_reward'])
                    max_rewards.append(step_data['max_reward'])
                    std_rewards.append(step_data.get('std_reward', 0))
        
        if not steps:
            warnings.warn("No step data found in history")
            return
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制曲线
        ax.plot(steps, mean_rewards, 'o-', label='Mean Reward', linewidth=2, markersize=6)
        ax.plot(steps, max_rewards, 's-', label='Max Reward', linewidth=2, markersize=6)
        
        # 绘制标准差区域
        mean_rewards = np.array(mean_rewards)
        std_rewards = np.array(std_rewards)
        ax.fill_between(
            steps, 
            mean_rewards - std_rewards, 
            mean_rewards + std_rewards,
            alpha=0.3,
            label='Mean ± Std'
        )
        
        # 设置标签
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Reward Progression During Training', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        sns.despine()
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"[TTTVisualizer] Saved progression plot to {output_path}")


def plot_training_dynamics(
    history_dict: dict,
    steps: list[int] = [0, 9, 24, 49],
    best_of_n_key: str = 'best_of_n',
    benchmark_value: float = 2.635983,
    benchmark_label: str = 'Best Human',
    xlabel: str = 'Sum of Radii (higher is better →)',
    ylabel: str = 'Probability Density',
    output_path: str = 'training_dynamics.png',
    higher_is_better: bool = True,
    reward_transform: Optional[callable] = None,
    xlim: Optional[tuple[float, float]] = None,
) -> None:
    """
    便捷的函数接口：绘制训练动态分布图
    
    Args:
        history_dict: 训练历史数据字典
        steps: 要绘制的 step 列表
        best_of_n_key: Best-of-N 数据的 key
        benchmark_value: Benchmark 阈值（如 Circle Packing n=26 SOTA）
        benchmark_label: Benchmark 标签
        xlabel: X 轴标签
        ylabel: Y 轴标签
        output_path: 输出图片路径
        higher_is_better: reward 是否越大越好
        reward_transform: 可选的 reward 转换函数，如 lambda r: 1/r 用于 AC1
        xlim: 可选的 x 轴显示范围元组 (min, max)
    """
    visualizer = TTTVisualizer(higher_is_better=higher_is_better)
    visualizer.plot_training_dynamics(
        history_dict=history_dict,
        steps=steps,
        best_of_n_key=best_of_n_key,
        benchmark_value=benchmark_value,
        benchmark_label=benchmark_label,
        xlabel=xlabel,
        ylabel=ylabel,
        output_path=output_path,
        reward_transform=reward_transform,
        xlim=xlim,
    )


def plot_reward_progression(
    history_dict: dict,
    output_path: str = 'reward_progression.png',
    reward_transform: Optional[callable] = None,
) -> None:
    """
    便捷的函数接口：绘制 reward 进展图
    
    Args:
        history_dict: 训练历史数据字典
        output_path: 输出图片路径
        reward_transform: 可选的 reward 转换函数，如 lambda r: 1/r 用于 AC1
    """
    visualizer = TTTVisualizer()
    visualizer.plot_reward_progression(history_dict, output_path, reward_transform=reward_transform)
