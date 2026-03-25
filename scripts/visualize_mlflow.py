#!/usr/bin/env python3
"""Visualize MLflow runs and compare versions."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available, using local artifacts only")

PROJECT_ROOT = Path(__file__).parent.parent


def plot_mlflow_comparison():
    """Compare multiple MLflow runs"""
    if not MLFLOW_AVAILABLE:
        print("Install mlflow: uv add mlflow")
        return
    
    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    
    # Get all runs
    runs = mlflow.search_runs(experiment_names=["kive_training"])
    
    if runs.empty:
        print("No MLflow runs found")
        return
    
    # Sort by start time
    runs = runs.sort_values("start_time")
    
    # Plot metrics over runs
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Reward progression
    if "metrics.final_reward" in runs.columns:
        ax1.plot(range(len(runs)), runs["metrics.final_reward"], 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel("Run Number")
        ax1.set_ylabel("Final Reward")
        ax1.set_title("Reward Improvement Across Runs")
        ax1.grid(alpha=0.3)
        ax1.axhline(0.75, color='green', linestyle='--', label='Target')
        ax1.legend()
    
    # FN rate progression
    if "metrics.fn_rate" in runs.columns:
        ax2.plot(range(len(runs)), runs["metrics.fn_rate"], 'o-', color='red', linewidth=2, markersize=8)
        ax2.set_xlabel("Run Number")
        ax2.set_ylabel("False Negative Rate")
        ax2.set_title("FN Rate Improvement Across Runs")
        ax2.grid(alpha=0.3)
        ax2.axhline(0.05, color='green', linestyle='--', label='Target < 5%')
        ax2.legend()
    
    # Probe usage progression
    if "metrics.mean_probes" in runs.columns:
        ax3.plot(range(len(runs)), runs["metrics.mean_probes"], 'o-', color='orange', linewidth=2, markersize=8)
        ax3.set_xlabel("Run Number")
        ax3.set_ylabel("Mean Probes per Episode")
        ax3.set_title("Probe Strategy Evolution")
        ax3.grid(alpha=0.3)
    
    # Episodes to convergence
    if "metrics.n_episodes" in runs.columns:
        ax4.bar(range(len(runs)), runs["metrics.n_episodes"], color='purple', alpha=0.7)
        ax4.set_xlabel("Run Number")
        ax4.set_ylabel("Episodes to Convergence")
        ax4.set_title("Training Efficiency")
        ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = PROJECT_ROOT / "docs/mlflow_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved MLflow comparison to {output_path}")
    
    # Print summary table
    print("\n=== MLflow Runs Summary ===")
    summary_cols = ["run_id", "metrics.final_reward", "metrics.fn_rate", "metrics.mean_probes", "metrics.n_episodes"]
    available_cols = [col for col in summary_cols if col in runs.columns]
    if available_cols:
        print(runs[available_cols].to_string(index=False))


def plot_local_artifacts():
    """Plot from local artifacts if MLflow not available"""
    learning_curve = PROJECT_ROOT / "artifacts/training/learning_curve.csv"
    
    if not learning_curve.exists():
        print("No training artifacts found")
        return
    
    df = pd.read_csv(learning_curve)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Reward over episodes
    ax1.plot(df['episode'], df['reward_mean'], linewidth=2)
    ax1.fill_between(df['episode'], 
                     df['reward_mean'] - df['reward_std'],
                     df['reward_mean'] + df['reward_std'],
                     alpha=0.3)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Reward Progression')
    ax1.axhline(0.75, color='green', linestyle='--', label='Target')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Error rates
    ax2.plot(df['episode'], df['fn_rate'], label='FN Rate', linewidth=2)
    ax2.plot(df['episode'], df['fp_rate'], label='FP Rate', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Error Rate')
    ax2.set_title('Error Rate Progression')
    ax2.axhline(0.05, color='green', linestyle='--', alpha=0.5)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Probe usage
    ax3.plot(df['episode'], df['probes_mean'], color='orange', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Mean Probes')
    ax3.set_title('Probe Strategy Evolution')
    ax3.grid(alpha=0.3)
    
    # Convergence metrics
    final_window = df.tail(10)
    metrics = {
        'Reward': final_window['reward_mean'].mean(),
        'FN Rate': final_window['fn_rate'].mean(),
        'FP Rate': final_window['fp_rate'].mean(),
        'Probes': final_window['probes_mean'].mean()
    }
    ax4.bar(metrics.keys(), metrics.values(), color=['green', 'red', 'orange', 'purple'], alpha=0.7)
    ax4.set_ylabel('Value')
    ax4.set_title('Final Performance Metrics')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = PROJECT_ROOT / "docs/training_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training summary to {output_path}")


if __name__ == "__main__":
    if MLFLOW_AVAILABLE:
        try:
            plot_mlflow_comparison()
        except Exception as e:
            print(f"MLflow plotting failed: {e}")
            print("Falling back to local artifacts...")
            plot_local_artifacts()
    else:
        plot_local_artifacts()
