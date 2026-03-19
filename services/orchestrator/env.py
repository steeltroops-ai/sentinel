# services/orchestrator/env.py
# ExpertFraudEnv — gymnasium.Env implementation for KIVE POMDP.
# Observation: 6D float32. Actions: PASS=0, REJECT=1, FLAG=2, PROBE=3.

from __future__ import annotations

import asyncio
from typing import Any, Optional

import gymnasium as gym
import numpy as np


class ExpertFraudEnv(gym.Env):
    """
    POMDP environment for expert fraud vetting with active probing.

    Observation space (6D, all normalized [0,1]):
        [fraud_belief, confidence, tav_score, svp_variance, fmd_score, norm_evidence_count]

    Action space — Discrete(4):
        0: PASS    — conclude expert is real, terminate
        1: REJECT  — conclude expert is fraud, terminate
        2: FLAG    — escalate to human reviewer, terminate
        3: PROBE   — generate targeted question, continue (max 5 probes)

    Reward (asymmetric by business cost):
        True Pass    = +1.0   (real expert correctly passed)
        True Reject  = +1.0   (fraud correctly rejected)
        False Neg    = -2.5   (fraud passed — platform damage)
        False Pos    = -1.0   (real expert rejected — opportunity cost)
        Flag Hit     = +0.3   (human reviewer correct)
        Flag Miss    = -0.2   (human reviewer wrong)
        Probe        = -0.1   (information acquisition cost per probe)
    """

    metadata = {"render_modes": []}

    # Reward constants — do not tune without strong justification
    R_TRUE_PASS   = +1.0
    R_TRUE_REJECT = +1.0
    R_FALSE_NEG   = -2.5   # Business critical: FN cost = 2.5x FP cost
    R_FALSE_POS   = -1.0
    R_FLAG_HIT    = +0.3
    R_FLAG_MISS   = -0.2
    R_PROBE       = -0.1

    MAX_PROBES = 5

    SIGNAL_WEIGHTS = {
        "tav": 0.28,
        "svp": 0.24,
        "fmd": 0.20,
        "mdc": 0.16,
        "tsi": 0.12,
    }

    def __init__(self, profile_generator, signal_client, render_mode=None):
        super().__init__()
        self.profile_generator = profile_generator
        self._signal_client_factory = signal_client  # keep original for re-seeding
        self.signal_client = signal_client

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)

        self._reset_state()

    def _reset_state(self):
        self.fraud_belief: float = 0.5
        self.confidence: float = 0.0
        self.tav_score: float = 0.5
        self.svp_variance: float = 0.5
        self.fmd_score: float = 0.5
        self.evidence_count: int = 0
        self._true_label: Optional[str] = None
        self._current_profile = None
        self._episode_reward: float = 0.0
        self._action_history: list[str] = []
        self._belief_history: list[float] = []

    def _obs(self) -> np.ndarray:
        return np.array([
            self.fraud_belief,
            self.confidence,
            self.tav_score,
            self.svp_variance,
            self.fmd_score,
            self.evidence_count / self.MAX_PROBES,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        # Re-seed the mock client if it supports it (gymnasium compliance requirement)
        if hasattr(self._signal_client_factory, '_rng') or hasattr(self._signal_client_factory, '__class__'):
            try:
                from services.orchestrator.signal_client import MockSignalClient
                if isinstance(self._signal_client_factory, MockSignalClient):
                    self.signal_client = MockSignalClient(rng=self.np_random)
            except ImportError:
                pass

        self._current_profile, self._true_label = self.profile_generator.sample(rng=self.np_random)

        # Extract all signals (synchronous wrapper over async signal calls)
        signals = self._extract_signals()
        self._update_from_signals(signals)
        self._belief_history.append(self.fraud_belief)

        info = {
            "true_label": self._true_label,
            "profile_id": getattr(self._current_profile, "id", "unknown"),
            "initial_signals": signals,
        }
        return self._obs(), info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        action_name = ["PASS", "REJECT", "FLAG", "PROBE"][action]

        # Force FLAG if probe limit reached
        if action == 3 and self.evidence_count >= self.MAX_PROBES:
            action = 2
            action_name = "FLAG"

        if action == 3:  # PROBE
            reward += self.R_PROBE
            probe_result = self._execute_probe()
            self.evidence_count += 1
            self._update_from_probe(probe_result)

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

    def _extract_signals(self) -> dict[str, dict]:
        """Call all signal services and return scores."""
        try:
            return asyncio.get_event_loop().run_until_complete(
                self.signal_client.extract_all(self._current_profile)
            )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            return loop.run_until_complete(
                self.signal_client.extract_all(self._current_profile)
            )

    def _update_from_signals(self, signals: dict[str, dict]):
        """Weighted aggregation of all signal scores into belief and confidence."""
        self.tav_score = float(signals.get("tav", {}).get("score", 0.5))
        self.svp_variance = float(signals.get("svp", {}).get("score", 0.5))
        self.fmd_score = float(signals.get("fmd", {}).get("score", 0.5))

        weighted_belief = (
            self.tav_score * self.SIGNAL_WEIGHTS["tav"] +
            self.svp_variance * self.SIGNAL_WEIGHTS["svp"] +
            self.fmd_score * self.SIGNAL_WEIGHTS["fmd"] +
            float(signals.get("mdc", {}).get("score", 0.5)) * self.SIGNAL_WEIGHTS["mdc"] +
            float(signals.get("tsi", {}).get("score", 0.5)) * self.SIGNAL_WEIGHTS["tsi"]
        )
        self.fraud_belief = float(np.clip(weighted_belief, 0.0, 1.0))

        all_scores = [
            self.tav_score, self.svp_variance, self.fmd_score,
            float(signals.get("mdc", {}).get("score", 0.5)),
            float(signals.get("tsi", {}).get("score", 0.5)),
        ]
        # Confidence: high when signals agree (low std dev)
        self.confidence = float(np.clip(1.0 - np.std(all_scores) * 2.0, 0.0, 1.0))

    def _execute_probe(self) -> dict:
        """Select the most ambiguous signal dimension and probe it."""
        ambiguity = {
            "tav": 1.0 - abs(self.tav_score - 0.5) * 2,
            "svp": 1.0 - abs(self.svp_variance - 0.5) * 2,
            "fmd": 1.0 - abs(self.fmd_score - 0.5) * 2,
        }
        target = max(ambiguity, key=ambiguity.get)

        try:
            return asyncio.get_event_loop().run_until_complete(
                self.signal_client.probe(self._current_profile, target, self.evidence_count)
            )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            return loop.run_until_complete(
                self.signal_client.probe(self._current_profile, target, self.evidence_count)
            )

    def _update_from_probe(self, probe_result: dict):
        """Bayesian update of fraud_belief from probe response."""
        score = float(probe_result.get("score", 0.5))
        weight = float(probe_result.get("weight", 0.20))

        prior = self.fraud_belief
        p_fraud = score
        p_real = 1.0 - score
        numerator = p_fraud * prior
        denominator = p_fraud * prior + p_real * (1.0 - prior) + 1e-8
        posterior = numerator / denominator

        self.fraud_belief = float(np.clip(prior + weight * (posterior - prior), 0.0, 1.0))
        # Each probe adds evidence, increasing confidence
        self.confidence = float(np.clip(self.confidence + 0.15, 0.0, 1.0))
