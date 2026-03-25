# services/orchestrator/train.py
# End-to-end RL training: synthetic data -> ExpertFraudEnv -> RecurrentPPO -> MLflow -> plots.
#
# Key design decisions:
#   - n_steps=128 calibrated for avg episode length of 3-5 steps
#   - ent_coef=0.05 for strong exploration early (decays via PPO clipping)
#   - gamma=0.99 standard temporal discount
#   - Probe counting fixed to track UNIQUE probes only (redundant probes penalized)
#   - Observation space reduced to 16D (removed redundant passive/active belief dims)
#   - Probe cost aligned to -0.05 (memo consistency)
#   - Active signal calibration reduced to force multi-probe strategy

from __future__ import annotations

import argparse
import csv
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
    n_episodes: int = 5000,
    eval_window: int = 100,
    run_name: str = "kive_ppo_v2",
    output_dir: str = "artifacts/training",
    use_mlflow: bool = True,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---- Setup ----
    print(f"Loading profiles from {profiles_path}")
    gen = ProfileGenerator.from_file(profiles_path)
    client = MockSignalClient()
    env = ExpertFraudEnv(gen, client)

    # Gymnasium compliance check
    try:
        from gymnasium.utils.env_checker import check_env
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check_env(env)
        print("gym.Env check: PASSED")
    except Exception as e:
        print(f"gym.Env check warning: {e}")

    # ---- MLflow ----
    run = None
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("kive_rl_training")
            run = mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "n_episodes": n_episodes,
                "agent": "RecurrentPPO",
                "max_probes": env.MAX_PROBES,
                "max_steps": env.MAX_STEPS,
                "reward_fn": "FN=-2.5,FP=-1.0,TP=+1.0,TN=+1.0,FLAG=+0.3/-0.2,PROBE=-0.05",
                "fraud_ratio": gen.fraud_ratio,
                "observation_dim": 16,
                "action_dim": 7,
            })
        except Exception as e:
            print(f"MLflow unavailable: {e}. Continuing without tracking.")
            use_mlflow = False

    # ---- Agent Setup ----
    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.callbacks import BaseCallback

        class KiveMetricsCallback(BaseCallback):
            def __init__(self, eval_window=100, use_mlflow=False):
                super().__init__(0)
                self.eval_window = eval_window
                self.use_mlflow = use_mlflow
                self.episodes = 0
                self.episode_rewards = []
                self.probe_counts = []
                self.fn_counts = []
                self.fp_counts = []
                self.traces = []
                
                # Stability tracking
                self.fn_rate_history = []
                self.fp_rate_history = []

            def _on_step(self) -> bool:
                dones = self.locals.get("dones", [False])
                if not dones[0]:
                    return True

                info = self.locals.get("infos", [{}])[0]
                self.episodes += 1
                self.episode_rewards.append(info.get("episode_reward", 0))

                history = info.get("action_history", [])

                # Count UNIQUE probes only (redundant probes are penalized but shouldn't count as valid evidence)
                unique_probes = set(a for a in history if a.startswith("PROBE_"))
                probe_count = len(unique_probes)
                self.probe_counts.append(probe_count)

                # FN/FP from terminal action
                true_label = info.get("true_label")
                last_action = history[-1] if history else "FLAG"
                fn = 1 if (last_action == "PASS" and true_label == "FRAUD") else 0
                fp = 1 if (last_action == "REJECT" and true_label == "REAL") else 0
                self.fn_counts.append(fn)
                self.fp_counts.append(fp)

                self.traces.append({
                    "true_label": true_label,
                    "beliefs": info.get("belief_history", []),
                    "actions": history,
                })
                if len(self.traces) > 20:
                    self.traces.pop(0)

                if self.episodes % self.eval_window == 0:
                    w = self.episode_rewards[-self.eval_window:]
                    pn = self.probe_counts[-self.eval_window:]
                    fn_r = sum(self.fn_counts[-self.eval_window:]) / self.eval_window
                    fp_r = sum(self.fp_counts[-self.eval_window:]) / self.eval_window
                    
                    # Track stability
                    self.fn_rate_history.append(fn_r)
                    self.fp_rate_history.append(fp_r)
                    
                    # Compute stability metrics (variance over last 5 windows)
                    if len(self.fn_rate_history) >= 5:
                        fn_stability = np.std(self.fn_rate_history[-5:])
                        fp_stability = np.std(self.fp_rate_history[-5:])
                    else:
                        fn_stability = 0.0
                        fp_stability = 0.0

                    metrics = {
                        "mean_reward": float(np.mean(w)),
                        "std_reward": float(np.std(w)),
                        "fn_rate": fn_r,
                        "fp_rate": fp_r,
                        "mean_probes": float(np.mean(pn)),
                        "fn_stability": fn_stability,
                        "fp_stability": fp_stability,
                    }
                    
                    # Detect policy collapse
                    collapse_warning = ""
                    if fn_r > 0.15:
                        collapse_warning = " [!!! FN COLLAPSE !!!]"
                    elif fp_r > 0.15:
                        collapse_warning = " [!!! FP COLLAPSE !!!]"
                    
                    print(
                        f"Ep {self.episodes:5d} | reward={metrics['mean_reward']:+.3f} "
                        f"| FN={metrics['fn_rate']:.3f} FP={metrics['fp_rate']:.3f} "
                        f"| probes={metrics['mean_probes']:.2f}{collapse_warning}"
                    )
                    if self.use_mlflow:
                        try:
                            import mlflow
                            mlflow.log_metrics(metrics, step=self.episodes)
                        except Exception:
                            pass
                return True

        # Hyperparameters calibrated for short-horizon POMDP
        #   n_steps=128:     ~25-40 episodes per rollout buffer
        #   batch_size=32:   small batches for high gradient signal
        #   ent_coef=0.05:   aggressive exploration (critical for probe discovery)
        #   gamma=0.99:      standard discount
        #   learning_rate=5e-4: slightly higher LR for faster initial learning
        #   n_epochs=8:      more passes over each batch
        agent = RecurrentPPO(
            "MlpLstmPolicy", env,
            verbose=0,
            n_steps=128,
            batch_size=32,
            n_epochs=8,
            gamma=0.99,
            learning_rate=5e-4,
            ent_coef=0.05,
            clip_range=0.2,
            max_grad_norm=0.5,
        )
        use_sb3 = True
    except Exception as e:
        print(f"SB3 setup failed: {e}")
        agent = None
        use_sb3 = False

    # ---- Training ----
    if use_sb3:
        # Estimate total timesteps: target episodes * avg_episode_length
        # With MAX_STEPS=8, avg episode ~4 steps after learning
        total_timesteps = n_episodes * 6
        print(f"Training RecurrentPPO agent ({total_timesteps} timesteps, ~{n_episodes} episodes)...")
        cb = KiveMetricsCallback(eval_window=eval_window, use_mlflow=use_mlflow)
        agent.learn(total_timesteps=total_timesteps, callback=cb)
        episode_rewards = cb.episode_rewards
        probe_counts = cb.probe_counts
        fn_counts = cb.fn_counts
        fp_counts = cb.fp_counts
        traces = cb.traces
    else:
        print("RecurrentPPO unavailable. Cannot continue.")
        return {}

    actual_episodes = len(episode_rewards)
    print(f"\nActual episodes completed: {actual_episodes}")

    # ---- Convergence report ----
    final_w = min(1000, actual_episodes)
    final_rewards = episode_rewards[-final_w:]
    final_probes = probe_counts[-final_w:]
    fn_r_final = sum(fn_counts[-final_w:]) / max(final_w, 1)
    fp_r_final = sum(fp_counts[-final_w:]) / max(final_w, 1)
    
    # Detect degenerate policy (too few probes or too uniform)
    probe_variance = float(np.var(final_probes))
    degenerate_policy = (
        float(np.mean(final_probes)) < 1.5  # Agent should probe 1.5+ times on average
        or probe_variance < 0.3  # Agent should vary probe count based on uncertainty
    )

    report = {
        "run_name": run_name,
        "n_episodes_target": n_episodes,
        "n_episodes_actual": actual_episodes,
        "final_mean_reward": round(float(np.mean(final_rewards)), 4),
        "final_std_reward": round(float(np.std(final_rewards)), 4),
        "fn_rate": round(fn_r_final, 4),
        "fp_rate": round(fp_r_final, 4),
        "mean_probes_per_episode": round(float(np.mean(final_probes)), 3),
        "probe_variance": round(probe_variance, 3),
        "degenerate_policy": degenerate_policy,
        "converged": (
            float(np.mean(final_rewards)) > 0.75
            and fn_r_final < 0.05
            and fp_r_final < 0.08
            and not degenerate_policy
        ),
    }

    report_path = Path(output_dir) / "convergence_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nConvergence report: {report}")

    # ---- Export CSVs ----
    csv_path = Path(output_dir) / "learning_curve.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward_mean", "reward_std", "fn_rate", "fp_rate", "probes_mean"])
        for i in range(actual_episodes // eval_window):
            idx = (i + 1) * eval_window
            w = episode_rewards[max(0, idx - eval_window):idx]
            fn_block = fn_counts[max(0, idx - eval_window):idx]
            fp_block = fp_counts[max(0, idx - eval_window):idx]
            pr_block = probe_counts[max(0, idx - eval_window):idx]
            writer.writerow([
                idx,
                round(float(np.mean(w)), 4),
                round(float(np.std(w)), 4),
                round(sum(fn_block) / eval_window, 4),
                round(sum(fp_block) / eval_window, 4),
                round(float(np.mean(pr_block)), 2),
            ])

    traces_path = Path(output_dir) / "sample_traces.csv"
    with open(traces_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "n_probes", "n_redundant", "actions", "final_action"])
        for tr in traces:
            acts = tr.get("actions", [])
            probe_actions = [a for a in acts if a.startswith("PROBE_")]
            n_probes = len(set(probe_actions))  # unique probes
            n_redundant = len(probe_actions) - n_probes  # redundant probe count
            writer.writerow([
                tr.get("true_label"),
                n_probes,
                n_redundant,
                " -> ".join(acts),
                acts[-1] if acts else "",
            ])

    # ---- Plots ----
    lc_path = _plot_learning_curve(episode_rewards, run_name, output_dir)
    tr_path = _plot_episode_traces(traces, output_dir)

    # ---- MLflow finalize ----
    if use_mlflow:
        try:
            import mlflow
            mlflow.log_artifact(str(report_path))
            mlflow.log_artifact(lc_path)
            mlflow.log_artifact(tr_path)
            mlflow.log_metrics({
                "final_mean_reward": report["final_mean_reward"],
                "final_fn_rate": report["fn_rate"],
                "final_fp_rate": report["fp_rate"],
                "final_mean_probes": report["mean_probes_per_episode"],
                "converged": float(report["converged"]),
            })
            if agent is not None:
                agent.save(str(Path(output_dir) / "ppo_policy"))
            mlflow.end_run()
        except Exception as e:
            print(f"MLflow finalization failed: {e}")

    print(f"\nTraining complete. Outputs in {output_dir}/")
    if report["converged"]:
        print("All convergence criteria MET.")
    else:
        print("Convergence criteria NOT met. Review reward function or hyperparameters.")
        if report.get("degenerate_policy"):
            print("WARNING: Degenerate policy detected (insufficient probing or no variance).")
            print(f"  Mean probes: {report['mean_probes_per_episode']:.2f} (target: >1.5)")
            print(f"  Probe variance: {report['probe_variance']:.2f} (target: >0.3)")
    return report


def _plot_learning_curve(rewards: list, title: str, output_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"KIVE RL Training -- {title}", fontsize=12, fontweight="bold")

    episodes = np.arange(1, len(rewards) + 1)
    arr = np.array(rewards)
    window = min(100, max(len(arr) // 10, 1))
    rolling = np.convolve(arr, np.ones(window) / window, mode="valid")

    ax = axes[0]
    ax.plot(episodes, arr, alpha=0.15, color="#4C9BE8", linewidth=0.5, label="Episode reward")
    ax.plot(
        episodes[window - 1:], rolling,
        color="#E84C4C", linewidth=2, label=f"Rolling mean (n={window})",
    )
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
    ax2.set_title(f"Final {len(final)} Episodes -- Reward Distribution")
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

    fig.suptitle("Episode Traces -- Belief Trajectory + Action Sequence", fontsize=11)

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
        ax.axhline(0.5, color="#888", linestyle="--", linewidth=0.8, alpha=0.5)
        final_action = actions[-1] if actions else "?"
        probe_actions = [a for a in actions if a.startswith("PROBE_")]
        n_probes = len(set(probe_actions))  # unique probes only
        n_redundant = len(probe_actions) - n_probes
        title_suffix = f"({n_probes}p" + (f"+{n_redundant}r)" if n_redundant > 0 else ")")
        ax.set_title(
            f"{label} -> {final_action} {title_suffix}",
            fontsize=8, color=color, fontweight="bold",
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel("Fraud Belief", fontsize=7)
        ax.tick_params(labelsize=7)

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
    parser.add_argument("--n-episodes", type=int, default=5000)
    parser.add_argument("--eval-window", type=int, default=100)
    parser.add_argument("--run-name", default="kive_ppo_v2")
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

    if report.get("converged"):
        print("\nAll convergence criteria MET. Ready to submit.")
    else:
        print("\nConvergence criteria NOT met. Debug reward function or training params.")
        sys.exit(1)
