#!/usr/bin/env python3
"""
TTT-Discover 训练动态可视化脚本

从保存的 training_history.pkl 文件生成论文风格的 KDE 分布图

Usage:
    # 基本用法（自动检测所有可用的 steps）
    python generate_plot.py --history_path ./outputs/training_history.pkl
    
    # 限制最多显示的 step 数量（均匀采样）
    python generate_plot.py --history_path ./outputs/training_history.pkl --max_steps 5
    
    # 指定特定 steps（覆盖自动检测）
    python generate_plot.py --history_path ./outputs/training_history.pkl --steps 0 9 24 49
    
    # 指定 benchmark 值（不同任务）
    python generate_plot.py --history_path ./outputs/training_history.pkl --benchmark_value 2.635983
    
    # 越小越好的任务（如 TriMul Runtime）
    python generate_plot.py --history_path ./outputs/training_history.pkl \
        --benchmark_value 100 \
        --xlabel "Runtime μs (lower is better ←)" \
        --higher_is_better False

    # AC1 任务（reward 存储的是 1/bound，需要转换回 bound）
    python generate_plot.py --history_path ./outputs/training_history.pkl \
        --reward_transform ac1 \
        --benchmark_value 1.5030 \
        --xlabel "Upper Bound (lower is better ←)" \
        --higher_is_better False \
        --benchmark_label "Literature SOTA"

    # 自定义输出路径
    python generate_plot.py --history_path ./outputs/training_history.pkl \
        --output_path ./figures/my_training.png
"""

import argparse
import pickle
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from areal.experimental.ttt_discover.ttt_visualizer import TTTVisualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TTT-Discover training dynamics visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect all steps from history (default behavior)
  python generate_plot.py --history_path ./outputs/training_history.pkl
  
  # Limit to 5 evenly-spaced steps
  python generate_plot.py --history_path ./outputs/training_history.pkl --max_steps 5
  
  # TriMul Runtime (smaller is better)
  python generate_plot.py --history_path ./outputs/training_history.pkl \\
      --benchmark_value 100 \\
      --xlabel "Runtime μs (lower is better ←)" \\
      --higher_is_better False
  
  # AC1 (AlphaEvolve) - reward stored as 1/bound
  python generate_plot.py --history_path ./outputs/training_history.pkl \\
      --reward_transform ac1 \\
      --benchmark_value 1.5030 \\
      --xlabel "Upper Bound (lower is better ←)" \\
      --higher_is_better False \\
      --benchmark_label "Literature SOTA"
  
  # Custom steps (overrides auto-detection)
  python generate_plot.py --history_path ./outputs/training_history.pkl \\
      --steps 0 10 20 30 40 \\
      --output_path ./figures/training_progress.png
        """
    )
    
    parser.add_argument(
        '--history_path',
        type=str,
        required=True,
        help='Path to training_history.pkl file'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=None,  # None means auto-detect from history
        help='Steps to visualize (default: auto-detect all available steps)'
    )
    
    parser.add_argument(
        '--max_steps',
        type=int,
        default=None,
        help='Maximum number of steps to plot. If history has more steps, '
             'evenly sample this many steps (default: plot all available steps)'
    )
    
    parser.add_argument(
        '--benchmark_value',
        type=float,
        default=2.635983,
        help='Benchmark threshold value (default: 2.635983 for Circle Packing n=26)'
    )
    
    parser.add_argument(
        '--benchmark_label',
        type=str,
        default='Best Human',
        help='Label for benchmark line (default: "Best Human")'
    )
    
    parser.add_argument(
        '--xlabel',
        type=str,
        default='Sum of Radii (higher is better →)',
        help='X-axis label'
    )
    
    parser.add_argument(
        '--ylabel',
        type=str,
        default='Probability Density',
        help='Y-axis label'
    )
    
    parser.add_argument(
        '--title',
        type=str,
        default='TTT-Discover Training Dynamics',
        help='Plot title'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default='training_dynamics.png',
        help='Output image path (default: training_dynamics.png)'
    )
    
    parser.add_argument(
        '--higher_is_better',
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=True,
        help='Whether higher reward is better (default: True)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Output image DPI (default: 300)'
    )
    
    parser.add_argument(
        '--reward_transform',
        type=str,
        default='none',
        choices=['none', 'reciprocal', 'ac1'],
        help='Transform applied to rewards before plotting. '
             '"reciprocal" or "ac1": plot 1/reward (for AC1 where reward=1/bound). '
             '"none": no transform (default: none)'
    )
    
    parser.add_argument(
        '--show_progression',
        action='store_true',
        help='Also generate reward progression plot'
    )
    
    parser.add_argument(
        '--progression_output',
        type=str,
        default='reward_progression.png',
        help='Output path for progression plot'
    )
    
    return parser.parse_args()


def load_history(path: str) -> dict:
    """加载训练历史文件"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_available_steps(history_dict: dict) -> list[int]:
    """
    从历史数据中提取所有可用的 step 编号
    
    Returns:
        按顺序排列的 step 列表
    """
    history = history_dict.get('history', history_dict)
    available_steps = set()
    
    for key in history.keys():
        if key.startswith('step_'):
            try:
                step = int(key.split('_')[1])
                available_steps.add(step)
            except (IndexError, ValueError):
                pass
    
    return sorted(available_steps)


def select_steps_evenly(all_steps: list[int], max_steps: int) -> list[int]:
    """
    从所有 steps 中均匀选择指定数量的 steps
    
    Args:
        all_steps: 所有可用的 steps
        max_steps: 最多选择多少个
        
    Returns:
        均匀采样的 steps 列表
    """
    if len(all_steps) <= max_steps:
        return all_steps
    
    # 均匀采样：包括第一个和最后一个，中间均匀分布
    indices = [int(i * (len(all_steps) - 1) / (max_steps - 1)) for i in range(max_steps)]
    return [all_steps[i] for i in indices]


def main():
    args = parse_args()
    
    # 检查文件是否存在
    history_path = Path(args.history_path)
    if not history_path.exists():
        print(f"Error: History file not found: {history_path}")
        sys.exit(1)
    
    # 加载历史数据
    print(f"Loading history from {history_path}...")
    history_dict = load_history(str(history_path))
    
    # 提取所有可用的 steps
    available_steps = get_available_steps(history_dict)
    
    if not available_steps:
        print("Error: No step data found in history file")
        print("Available keys:", list(history_dict.get('history', history_dict).keys())[:10])
        sys.exit(1)
    
    # 打印摘要信息
    metadata = history_dict.get('metadata', {})
    print("\n" + "="*60)
    print("Training History Summary")
    print("="*60)
    print(f"Total steps recorded: {len(available_steps)}")
    print(f"Step range: {min(available_steps)} - {max(available_steps)}")
    print(f"Available steps: {available_steps}")
    print(f"Overall best reward: {metadata.get('overall_best_reward', 'N/A')}")
    print(f"Overall best step: {metadata.get('overall_best_step', 'N/A')}")
    print(f"Has Best-of-N baseline: {'best_of_n' in history_dict}")
    print("="*60 + "\n")
    
    # 确定要绘制的 steps
    if args.steps is not None:
        # 用户指定了 steps
        requested_steps = set(args.steps)
        valid_steps = sorted(requested_steps & set(available_steps))
        missing_steps = sorted(requested_steps - set(available_steps))
        
        if missing_steps:
            print(f"Warning: Steps not found in history and will be skipped: {missing_steps}")
        
        if not valid_steps:
            print(f"Error: None of the requested steps {args.steps} found in history")
            print(f"Available steps: {available_steps}")
            sys.exit(1)
        
        print(f"Using user-specified steps: {valid_steps}")
    else:
        # 自动检测 steps
        if args.max_steps is not None and len(available_steps) > args.max_steps:
            valid_steps = select_steps_evenly(available_steps, args.max_steps)
            print(f"Auto-selected {len(valid_steps)} steps from {len(available_steps)} available:")
            print(f"  All available: {available_steps}")
            print(f"  Selected: {valid_steps}")
        else:
            valid_steps = available_steps
            print(f"Using all {len(valid_steps)} available steps: {valid_steps}")
    
    # 确定 reward transform
    reward_transform = None
    if args.reward_transform in ['reciprocal', 'ac1']:
        reward_transform = lambda r: 1.0 / (r + 1e-10)  # 避免除以零
        print(f"\n[AC1 Mode] Using reward transform: 1/reward (converting to bound)")
        print(f"  Original reward: 1/bound -> Plotting bound directly")
    
    # 创建可视化器
    visualizer = TTTVisualizer(higher_is_better=args.higher_is_better)
    
    # 确保输出目录存在
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 生成主图
    print(f"\nGenerating training dynamics plot...")
    visualizer.plot_training_dynamics(
        history_dict=history_dict,
        steps=valid_steps,
        benchmark_value=args.benchmark_value,
        benchmark_label=args.benchmark_label,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        title=args.title,
        output_path=str(output_path),
        dpi=args.dpi,
        reward_transform=reward_transform,
    )
    print(f"Saved to: {output_path.absolute()}")
    
    # 生成进展图
    if args.show_progression:
        print(f"\nGenerating reward progression plot...")
        progression_path = Path(args.progression_output)
        progression_path.parent.mkdir(parents=True, exist_ok=True)
        visualizer.plot_reward_progression(
            history_dict=history_dict,
            output_path=str(progression_path),
            dpi=args.dpi,
            reward_transform=reward_transform,
        )
        print(f"Saved to: {progression_path.absolute()}")
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)
    print("\nHow to interpret the plot:")
    print("-" * 40)
    print("1. Distribution shift: The blue curves should shift towards")
    if args.higher_is_better:
        print("   the RIGHT (higher reward) as training progresses.")
    else:
        print("   the LEFT (lower reward) as training progresses.")
    print("2. Peak narrowing: A sharper peak indicates more consistent")
    print("   performance from the policy.")
    print("3. Benchmark comparison: The black dashed line shows the")
    print("   human expert performance.")
    print("4. Best-of-N baseline: Gray curve shows the frozen policy")
    print("   performance (if available).")
    print("-" * 40)


if __name__ == "__main__":
    main()
