# services/orchestrator/env.py
# ExpertFraudEnv — gymnasium.Env for KIVE POMDP.
#
# Architecture: Passive/Active signal split.
#   - Passive signals (TAV, SVP, FMD, MDC, TSI) are visible at reset.
#   - Active signals (BES, LQA, CCS, RSL) require explicit probe actions.
#
# Observation (18D):
#   [0]  fraud_belief          — Bayesian posterior (agent-maintained)
#   [1]  confidence            — agreement across acquired signals
#   [2]  tav_score             — passive: temporal anchoring violations
#   [3]  svp_score             — passive: specificity variance profile
#   [4]  fmd_score             — passive: failure memory deficiency
#   [5]  mdc_score             — passive: market demand correlation
#   [6]  tsi_score             — passive: trajectory smoothness index
#   [7]  bes_score             — active: behavioral entropy (probe required)
#   [8]  lqa_score             — active: linguistic quality assurance (probe required)
#   [9]  ccs_score             — active: cross-candidate similarity (probe required)
#   [10] rsl_score             — active: response latency slope (probe required)
#   [11] evidence_count_norm   — probes used / MAX_PROBES
#   [12] probed_bes            — binary: has BES been probed?
#   [13] probed_lqa            — binary: has LQA been probed?
#   [14] probed_ccs            — binary: has CCS been probed?
#   [15] probed_rsl            — binary: has RSL been probed?
#   [16] passive_belief        — weighted belief from passive-only signals
#   [17] active_belief         — weighted belief from active-only signals (0.5 if unprobed)
#
# Action space — Discrete(7):
#   0: PASS                — conclude real, terminate
#   1: REJECT              — conclude fraud, terminate
#   2: FLAG                — escalate to human, terminate
#   3: PROBE_BES           — acquire behavioral entropy signal
#   4: PROBE_LQA           — acquire linguistic quality signal
#   5: PROBE_CCS           — acquire cross-candidate similarity
#   6: PROBE_RSL           — acquire response latency slope

from __future__ import annotations

import asyncio
from typing import Any, Optional

import gymnasium as gym
import numpy as np


# Signal taxonomy
PASSIVE_SIGNALS = ("tav", "svp", "fmd", "mdc", "tsi")
ACTIVE_SIGNALS  = ("bes", "lqa", "ccs", "rsl")
ALL_SIGNALS     = PASSIVE_SIGNALS + ACTIVE_SIGNALS

SIGNAL_WEIGHTS = {
    "tav": 0.14, "svp": 0.11, "fmd": 0.11, "mdc": 0.09, "tsi": 0.07,
    "bes": 0.18, "lqa": 0.12, "ccs": 0.10, "rsl": 0.08,
}

ACTION_NAMES = [
    "PASS", "REJECT", "FLAG",
    "PROBE_BES", "PROBE_LQA", "PROBE_CCS", "PROBE_RSL",
]

# Action -> active signal mapping
PROBE_ACTION_MAP = {3: "bes", 4: "lqa", 5: "ccs", 6: "rsl"}


class ExpertFraudEnv(gym.Env):
    """
    KIVE POMDP environment for expert fraud vetting with active probing.

    The agent receives passive signals for free (resume analysis) and must
    decide whether to spend probe budget acquiring active signals (live
    behavioral interaction) before issuing a terminal verdict.

    Reward (asymmetric by business cost):
        True Pass    = +1.0   (real expert correctly passed)
        True Reject  = +1.0   (fraud correctly rejected)
        False Neg    = -2.5   (fraud passed -- platform damage)
        False Pos    = -1.0   (real expert rejected -- opportunity cost)
        Flag Hit     = +0.3   (human reviewer correct)
        Flag Miss    = -0.2   (human reviewer wrong)
        Probe        = -0.05  (information acquisition cost per probe)
        Redundant    = -0.15  (re-probing already-acquired signal)
    """

    metadata = {"render_modes": []}

    R_TRUE_PASS   = +1.0
    R_TRUE_REJECT = +1.0
    R_FALSE_NEG   = -2.5
    R_FALSE_POS   = -1.0
    R_FLAG_HIT    = +0.3
    R_FLAG_MISS   = -0.2
    R_PROBE       = -0.02   # Low cost: probing should be encouraged
    R_REDUNDANT   = -0.20   # Harsh: don't waste time re-probing

    MAX_PROBES    = 4       # exactly 4 active signals to acquire
    MAX_STEPS     = 10      # room for full probe sequence + decision

    def __init__(self, profile_generator, signal_client, render_mode=None):
        super().__init__()
        self.profile_generator = profile_generator
        self._signal_client_factory = signal_client
        self.signal_client = signal_client

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(18,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(7)

        self._reset_state()

    def _reset_state(self):
        # Belief tracking
        self.fraud_belief: float = 0.5
        self.confidence: float = 0.0

        # Passive signal scores (populated at reset)
        self.tav_score: float = 0.5
        self.svp_score: float = 0.5
        self.fmd_score: float = 0.5
        self.mdc_score: float = 0.5
        self.tsi_score: float = 0.5

        # Active signal scores (require probing)
        self.bes_score: float = 0.5
        self.lqa_score: float = 0.5
        self.ccs_score: float = 0.5
        self.rsl_score: float = 0.5

        # Probe tracking
        self.probed: dict[str, bool] = {s: False for s in ACTIVE_SIGNALS}
        self.evidence_count: int = 0
        self.step_count: int = 0

        # Episode tracking
        self._true_label: Optional[str] = None
        self._current_profile = None
        self._episode_reward: float = 0.0
        self._action_history: list[str] = []
        self._belief_history: list[float] = []

    def _obs(self) -> np.ndarray:
        """Construct 18D observation. Every dimension is unique and meaningful."""
        passive_belief = self._compute_partial_belief(PASSIVE_SIGNALS)
        active_belief  = self._compute_partial_belief(ACTIVE_SIGNALS)

        return np.array([
            self.fraud_belief,                          # 0  aggregate belief
            self.confidence,                            # 1  signal agreement
            self.tav_score,                             # 2  passive
            self.svp_score,                             # 3  passive
            self.fmd_score,                             # 4  passive
            self.mdc_score,                             # 5  passive
            self.tsi_score,                             # 6  passive
            self.bes_score,                             # 7  active (0.5 = unprobed)
            self.lqa_score,                             # 8  active (0.5 = unprobed)
            self.ccs_score,                             # 9  active (0.5 = unprobed)
            self.rsl_score,                             # 10 active (0.5 = unprobed)
            self.evidence_count / self.MAX_PROBES,      # 11 probe budget usage
            float(self.probed["bes"]),                  # 12 probed flag
            float(self.probed["lqa"]),                  # 13 probed flag
            float(self.probed["ccs"]),                  # 14 probed flag
            float(self.probed["rsl"]),                  # 15 probed flag
            passive_belief,                             # 16 passive-only belief
            active_belief,                              # 17 active-only belief
        ], dtype=np.float32)

    def _compute_partial_belief(self, signal_set: tuple[str, ...]) -> float:
        """Weighted belief from a subset of signals."""
        scores = {
            "tav": self.tav_score, "svp": self.svp_score,
            "fmd": self.fmd_score, "mdc": self.mdc_score,
            "tsi": self.tsi_score, "bes": self.bes_score,
            "lqa": self.lqa_score, "ccs": self.ccs_score,
            "rsl": self.rsl_score,
        }
        total_w = sum(SIGNAL_WEIGHTS[s] for s in signal_set)
        if total_w < 1e-8:
            return 0.5
        belief = sum(scores[s] * SIGNAL_WEIGHTS[s] for s in signal_set) / total_w
        return float(np.clip(belief, 0.0, 1.0))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        # Re-seed mock client for gymnasium determinism compliance
        try:
            from services.orchestrator.signal_client import MockSignalClient
            if isinstance(self._signal_client_factory, MockSignalClient):
                self.signal_client = MockSignalClient(rng=self.np_random)
        except ImportError:
            pass

        self._current_profile, self._true_label = self.profile_generator.sample(
            rng=self.np_random
        )

        # Extract PASSIVE signals only -- these are free (resume analysis)
        signals = self._extract_signals()
        self._update_passive_signals(signals)

        # Compute initial belief from passive signals only
        self.fraud_belief = self._compute_partial_belief(PASSIVE_SIGNALS)
        self.confidence = self._compute_signal_agreement()

        self._belief_history.append(self.fraud_belief)

        info = {
            "true_label": self._true_label,
            "profile_id": getattr(self._current_profile, "id", "unknown"),
        }
        return self._obs(), info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        self.step_count += 1

        action_name = ACTION_NAMES[action]

        if action in PROBE_ACTION_MAP:
            # ---- PROBE action ----
            target = PROBE_ACTION_MAP[action]

            if self.probed[target]:
                # Penalize redundant probing -- agent should learn not to repeat
                reward += self.R_REDUNDANT
            else:
                reward += self.R_PROBE
                probe_result = self._execute_probe(target)
                self._update_active_signal(target, probe_result)
                self.probed[target] = True
                self.evidence_count += 1

                # Recompute aggregate belief with new evidence
                self._recompute_belief()

        elif action == 0:  # PASS
            terminated = True
            if self._true_label == "REAL":
                reward += self.R_TRUE_PASS
            else:
                reward += self.R_FALSE_NEG

        elif action == 1:  # REJECT
            terminated = True
            if self._true_label == "FRAUD":
                reward += self.R_TRUE_REJECT
            else:
                reward += self.R_FALSE_POS

        elif action == 2:  # FLAG
            terminated = True
            human_correct = self.np_random.random() < 0.70
            reward += self.R_FLAG_HIT if human_correct else self.R_FLAG_MISS
            info["human_reviewed"] = True
            info["human_correct"] = human_correct

        # ---- Truncation guard ----
        if not terminated and self.step_count >= self.MAX_STEPS:
            truncated = True
            # Force FLAG on timeout
            human_correct = self.np_random.random() < 0.70
            reward += self.R_FLAG_HIT if human_correct else self.R_FLAG_MISS
            action_name = "FLAG_TIMEOUT"

        self._episode_reward += reward
        self._action_history.append(action_name)
        self._belief_history.append(self.fraud_belief)

        info.update({
            "action": action_name,
            "fraud_belief": self.fraud_belief,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "true_label": self._true_label,
            "step_reward": reward,
            "episode_reward": self._episode_reward,
            "action_history": list(self._action_history),
            "belief_history": list(self._belief_history),
        })

        return self._obs(), reward, terminated, truncated, info

    # ---- Signal extraction ----

    def _extract_signals(self) -> dict[str, dict]:
        """Call all signal services and return raw scores."""
        try:
            return asyncio.run(self.signal_client.extract_all(self._current_profile))
        except RuntimeError:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(
                self.signal_client.extract_all(self._current_profile)
            )

    def _update_passive_signals(self, signals: dict[str, dict]):
        """Only populate passive signal slots. Active slots remain at 0.5."""
        self.tav_score = float(signals.get("tav", {}).get("score", 0.5))
        self.svp_score = float(signals.get("svp", {}).get("score", 0.5))
        self.fmd_score = float(signals.get("fmd", {}).get("score", 0.5))
        self.mdc_score = float(signals.get("mdc", {}).get("score", 0.5))
        self.tsi_score = float(signals.get("tsi", {}).get("score", 0.5))

    def _execute_probe(self, target: str) -> dict:
        """Execute a targeted probe via the signal client."""
        try:
            return asyncio.run(
                self.signal_client.probe(self._current_profile, target, self.evidence_count)
            )
        except RuntimeError:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(
                self.signal_client.probe(self._current_profile, target, self.evidence_count)
            )

    def _update_active_signal(self, target: str, probe_result: dict):
        """Update the specific active signal dimension from probe result."""
        score = float(probe_result.get("score", 0.5))
        score = float(np.clip(score, 0.0, 1.0))

        if target == "bes":
            self.bes_score = score
        elif target == "lqa":
            self.lqa_score = score
        elif target == "ccs":
            self.ccs_score = score
        elif target == "rsl":
            self.rsl_score = score

    def _recompute_belief(self):
        """Full Bayesian recomputation of fraud_belief from ALL acquired evidence."""
        all_scores = []
        all_weights = []

        # Always include passive signals
        for sig in PASSIVE_SIGNALS:
            score = getattr(self, f"{sig}_score", 0.5)
            all_scores.append(score)
            all_weights.append(SIGNAL_WEIGHTS[sig])

        # Include active signals ONLY if probed
        for sig in ACTIVE_SIGNALS:
            if self.probed[sig]:
                score = getattr(self, f"{sig}_score", 0.5)
                all_scores.append(score)
                all_weights.append(SIGNAL_WEIGHTS[sig])

        total_w = sum(all_weights)
        if total_w < 1e-8:
            self.fraud_belief = 0.5
        else:
            self.fraud_belief = float(np.clip(
                sum(s * w for s, w in zip(all_scores, all_weights)) / total_w,
                0.0, 1.0
            ))

        self.confidence = self._compute_signal_agreement()

    def _compute_signal_agreement(self) -> float:
        """Confidence = 1 - 2*std(acquired_scores). High when signals agree."""
        acquired = []
        for sig in PASSIVE_SIGNALS:
            acquired.append(getattr(self, f"{sig}_score", 0.5))
        for sig in ACTIVE_SIGNALS:
            if self.probed[sig]:
                acquired.append(getattr(self, f"{sig}_score", 0.5))

        if len(acquired) < 2:
            return 0.0
        return float(np.clip(1.0 - np.std(acquired) * 2.0, 0.0, 1.0))
