#!/usr/bin/env python3
"""
TTT-Discover 训练动态可视化脚本

从保存的 training_history.pkl 文件生成论文风格的 KDE 分布图

Usage:
    # 基本用法（Circle Packing 任务）
    python generate_plot.py --history_path ./outputs/training_history.pkl
    
    # 指定特定步骤
    python generate_plot.py --history_path ./outputs/training_history.pkl --steps 0 9 24 49
    
    # 指定 benchmark 值（不同任务）
    python generate_plot.py --history_path ./outputs/training_history.pkl --benchmark_value 2.635983
    
    # 越小越好的任务（如 TriMul Runtime）
    python generate_plot.py --history_path ./outputs/training_history.pkl \
        --benchmark_value 100 \
        --xlabel "Runtime μs (lower is better ←)" \
        --higher_is_better False

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
  # Circle Packing (default)
  python generate_plot.py --history_path ./outputs/training_history.pkl
  
  # TriMul Runtime (smaller is better)
  python generate_plot.py --history_path ./outputs/training_history.pkl \\
      --benchmark_value 100 \\
      --xlabel "Runtime μs (lower is better ←)" \\
      --higher_is_better False
  
  # Custom steps and output
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
        default=[0, 9, 24, 49],
        help='Steps to visualize (default: 0 9 24 49)'
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
    
    # 打印摘要信息
    metadata = history_dict.get('metadata', {})
    print("\n" + "="*60)
    print("Training History Summary")
    print("="*60)
    print(f"Number of snapshots: {metadata.get('num_snapshots', 'N/A')}")
    print(f"Save steps: {metadata.get('save_steps', 'N/A')}")
    print(f"Overall best reward: {metadata.get('overall_best_reward', 'N/A')}")
    print(f"Overall best step: {metadata.get('overall_best_step', 'N/A')}")
    print(f"Has Best-of-N baseline: {'best_of_n' in history_dict}")
    print("="*60 + "\n")
    
    # 过滤只存在于历史中的 steps
    history = history_dict.get('history', history_dict)
    available_steps = set()
    for key in history.keys():
        if key.startswith('step_'):
            try:
                step = int(key.split('_')[1])
                available_steps.add(step)
            except (IndexError, ValueError):
                pass
    
    requested_steps = set(args.steps)
    valid_steps = sorted(requested_steps & available_steps)
    missing_steps = sorted(requested_steps - available_steps)
    
    if missing_steps:
        print(f"Warning: Steps not found in history and will be skipped: {missing_steps}")
    
    if not valid_steps:
        print(f"Error: None of the requested steps {args.steps} found in history")
        print(f"Available steps: {sorted(available_steps)}")
        sys.exit(1)
    
    print(f"Generating plot for steps: {valid_steps}")
    
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
