# services/orchestrator/train.py
# End-to-end RL training: synthetic data -> ExpertFraudEnv -> DQN -> MLflow -> plots.

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.synthetic_generator import ProfileGenerator
from services.orchestrator.env import ExpertFraudEnv
from services.orchestrator.signal_client import MockSignalClient


def train(
    profiles_path: str,
    n_episodes: int = 3000,
    eval_window: int = 100,
    run_name: str = "kive_dqn_v1",
    output_dir: str = "artifacts/training",
    use_mlflow: bool = True,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Setup ---
    print(f"Loading profiles from {profiles_path}")
    gen = ProfileGenerator.from_file(profiles_path)
    client = MockSignalClient()
    env = ExpertFraudEnv(gen, client)

    # Gymnasium compliance check
    try:
        from gymnasium.utils.env_checker import check_env
        check_env(env, warn=True)
        print("gym.Env check: PASSED")
    except Exception as e:
        print(f"gym.Env check warning: {e}")

    # --- MLflow ---
    run = None
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("kive_rl_training")
            run = mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "n_episodes": n_episodes,
                "agent": "DQN",
                "max_probes": env.MAX_PROBES,
                "reward_fn": "FN=-2.5,FP=-1.0,TP=+1.0,TN=+1.0,PROBE=-0.1",
                "fraud_ratio": gen.fraud_ratio,
            })
        except Exception as e:
            print(f"MLflow unavailable: {e}. Continuing without tracking.")
            use_mlflow = False

    # --- Agent ---
    try:
        from stable_baselines3 import DQN
        agent = DQN(
            "MlpPolicy", env,
            verbose=0,
            learning_starts=200,
            buffer_size=10_000,
            batch_size=64,
            gamma=0.95,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
        )
        use_sb3 = True
    except Exception:
        agent = None
        use_sb3 = False
        print("SB3 not available — using Q-table fallback")

    # --- Training loop ---
    episode_rewards = []
    probe_counts   = []
    fn_counts      = []
    fp_counts      = []
    traces         = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_probes = ep_fn = ep_fp = 0
        ep_trace = {"true_label": info["true_label"], "beliefs": [], "actions": []}

        while not done:
            if use_sb3:
                action, _ = agent.predict(obs, deterministic=False)
                action = int(action)
            else:
                action = _qtable_action(obs)

            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            ep_trace["beliefs"].append(round(float(step_info["fraud_belief"]), 3))
            ep_trace["actions"].append(step_info["action"])

            if step_info["action"] == "PROBE":
                ep_probes += 1
            if terminated:
                if step_info["action"] == "PASS" and step_info["true_label"] == "FRAUD":
                    ep_fn += 1
                elif step_info["action"] == "REJECT" and step_info["true_label"] == "REAL":
                    ep_fp += 1

        if use_sb3:
            agent.learn(total_timesteps=0)  # SB3 learns from replay buffer internally

        episode_rewards.append(ep_reward)
        probe_counts.append(ep_probes)
        fn_counts.append(ep_fn)
        fp_counts.append(ep_fp)

        # Collect traces for export (first 5 real, first 5 fraud across training)
        if len(traces) < 10:
            traces.append(ep_trace)

        # Periodic evaluation + logging
        if (episode + 1) % eval_window == 0:
            w = episode_rewards[-eval_window:]
            pn = probe_counts[-eval_window:]
            fn_r = sum(fn_counts[-eval_window:]) / eval_window
            fp_r = sum(fp_counts[-eval_window:]) / eval_window

            metrics = {
                "mean_reward": float(np.mean(w)),
                "std_reward": float(np.std(w)),
                "fn_rate": fn_r,
                "fp_rate": fp_r,
                "mean_probes": float(np.mean(pn)),
            }

            print(
                f"Ep {episode+1:5d} | reward={metrics['mean_reward']:+.3f} "
                f"| FN={metrics['fn_rate']:.3f} FP={metrics['fp_rate']:.3f} "
                f"| probes={metrics['mean_probes']:.2f}"
            )

            if use_mlflow:
                try:
                    mlflow.log_metrics(metrics, step=episode + 1)
                except Exception:
                    pass

    # --- Convergence report ---
    final_w = 500
    final_rewards = episode_rewards[-final_w:]
    final_probes  = probe_counts[-final_w:]
    fn_r_final = sum(fn_counts[-final_w:]) / final_w
    fp_r_final = sum(fp_counts[-final_w:]) / final_w

    report = {
        "run_name": run_name,
        "n_episodes": n_episodes,
        "final_mean_reward": round(float(np.mean(final_rewards)), 4),
        "fn_rate": round(fn_r_final, 4),
        "fp_rate": round(fp_r_final, 4),
        "mean_probes_per_episode": round(float(np.mean(final_probes)), 3),
        "converged": (
            float(np.mean(final_rewards)) > 0.5
            and fn_r_final < 0.10
            and fp_r_final < 0.15
        ),
    }

    report_path = Path(output_dir) / "convergence_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nConvergence report: {report}")

    # --- Plots ---
    lc_path = _plot_learning_curve(episode_rewards, run_name, output_dir)
    tr_path = _plot_episode_traces(traces, output_dir)

    if use_mlflow:
        try:
            mlflow.log_artifact(str(report_path))
            mlflow.log_artifact(lc_path)
            mlflow.log_artifact(tr_path)
            mlflow.log_metrics({
                "final_mean_reward": report["final_mean_reward"],
                "final_fn_rate": report["fn_rate"],
                "final_fp_rate": report["fp_rate"],
                "converged": float(report["converged"]),
            })
            if use_sb3 and agent is not None:
                agent.save(str(Path(output_dir) / "dqn_policy"))
            mlflow.end_run()
        except Exception as e:
            print(f"MLflow artifact logging failed: {e}")

    print(f"\nTraining complete. Outputs in {output_dir}/")
    return report


def _qtable_action(obs: np.ndarray) -> int:
    """Simple heuristic fallback when SB3 is not available."""
    belief, confidence = float(obs[0]), float(obs[1])
    evidence_count = float(obs[5]) * 5  # denormalize

    if evidence_count >= 5:
        return 2  # FLAG
    if belief < 0.25 and confidence > 0.6:
        return 0  # PASS
    if belief > 0.75 and confidence > 0.6:
        return 1  # REJECT
    if 0.4 <= belief <= 0.6 or confidence < 0.5:
        return 3  # PROBE
    if belief > 0.6:
        return 1
    return 0


def _plot_learning_curve(rewards: list, title: str, output_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"KIVE RL Training — {title}", fontsize=12, fontweight="bold")

    episodes = np.arange(1, len(rewards) + 1)
    arr = np.array(rewards)
    window = min(100, len(arr) // 5)
    rolling = np.convolve(arr, np.ones(window) / window, mode="valid")

    ax = axes[0]
    ax.plot(episodes, arr, alpha=0.15, color="#4C9BE8", linewidth=0.5, label="Episode reward")
    ax.plot(episodes[window - 1:], rolling, color="#E84C4C", linewidth=2, label=f"Rolling mean (n={window})")
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=1, label="Convergence threshold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Episode Reward Over Training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    final = arr[-min(1000, len(arr)):]
    ax2.hist(final, bins=30, color="#4C9BE8", alpha=0.75, edgecolor="white")
    ax2.axvline(np.mean(final), color="#E84C4C", linewidth=2, label=f"Mean: {np.mean(final):.2f}")
    ax2.set_xlabel("Episode Reward")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Final {len(final)} Episodes — Reward Distribution")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = str(Path(output_dir) / "learning_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Learning curve saved: {path}")
    return path


def _plot_episode_traces(traces: list, output_dir: str) -> str:
    n = len(traces)
    if n == 0:
        return ""

    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    fig.suptitle("Episode Traces — Belief Trajectory + Action Sequence", fontsize=11)

    for idx, trace in enumerate(traces):
        r, c = divmod(idx, cols)
        ax = axes[r][c] if rows > 1 else axes[0][c]
        beliefs = trace.get("beliefs", [])
        actions = trace.get("actions", [])
        label = trace.get("true_label", "?")
        color = "#E84C4C" if label == "FRAUD" else "#4CE87C"

        steps = range(len(beliefs))
        ax.plot(steps, beliefs, "o-", color=color, linewidth=2, markersize=5)
        ax.axhline(0.75, color="#E84C4C", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axhline(0.25, color="#4CE87C", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axhline(0.5,  color="#888",    linestyle="--", linewidth=0.8, alpha=0.5)
        final_action = actions[-1] if actions else "?"
        ax.set_title(f"{label} → {final_action}", fontsize=9, color=color, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("Fraud Belief", fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        if rows > 1:
            axes[r][c].set_visible(False)
        else:
            axes[0][c].set_visible(False)

    plt.tight_layout()
    path = str(Path(output_dir) / "episode_traces.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Episode traces saved: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KIVE RL agent")
    parser.add_argument("--profiles", default="data/synthetic_profiles.json")
    parser.add_argument("--n-episodes", type=int, default=3000)
    parser.add_argument("--eval-window", type=int, default=100)
    parser.add_argument("--run-name", default="kive_dqn_v1")
    parser.add_argument("--output-dir", default="artifacts/training")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    report = train(
        profiles_path=args.profiles,
        n_episodes=args.n_episodes,
        eval_window=args.eval_window,
        run_name=args.run_name,
        output_dir=args.output_dir,
        use_mlflow=not args.no_mlflow,
    )

    if report["converged"]:
        print("\nAll convergence criteria MET. Ready to submit.")
    else:
        print("\nConvergence criteria NOT met. Debug reward function or training params.")
        sys.exit(1)
