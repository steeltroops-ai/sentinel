import pytest
import numpy as np
from services.orchestrator.env import ExpertFraudEnv
from services.orchestrator.signal_client import MockSignalClient
from data.synthetic_generator import ProfileGenerator, RealExpertGenerator, FraudExpertGenerator


def _make_env():
    gen = ProfileGenerator(fraud_ratio=0.5)
    client = MockSignalClient()
    return ExpertFraudEnv(gen, client)


def test_observation_space():
    env = _make_env()
    obs, _ = env.reset()
    assert obs.shape == (6,), f"Expected (6,), got {obs.shape}"
    assert obs.dtype == np.float32
    assert all(0.0 <= v <= 1.0 for v in obs), f"Obs out of [0,1]: {obs}"


def test_probe_does_not_terminate():
    env = _make_env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(3)  # PROBE
    assert not terminated
    assert reward == pytest.approx(-0.1, abs=1e-6)
    assert info["action"] == "PROBE"
    assert info["evidence_count"] == 1


def test_probe_limit_forces_flag():
    env = _make_env()
    env.reset()
    # Max out probes
    for _ in range(env.MAX_PROBES):
        obs, _, terminated, _, _ = env.step(3)
        if terminated:
            break
    # Next PROBE should auto-convert to FLAG
    obs, reward, terminated, truncated, info = env.step(3)
    assert terminated
    assert info["action"] == "FLAG"


def test_false_negative_penalty():
    """PASS on a FRAUD profile must return -2.5."""
    client = MockSignalClient()
    # Force a fraud profile
    gen = ProfileGenerator(fraud_ratio=1.0)
    env = ExpertFraudEnv(gen, client)
    env.reset()
    env._true_label = "FRAUD"
    _, reward, terminated, _, info = env.step(0)  # PASS
    assert terminated
    assert reward == pytest.approx(-2.5, abs=1e-6)


def test_true_reject_reward():
    """REJECT on a FRAUD profile must return +1.0."""
    client = MockSignalClient()
    gen = ProfileGenerator(fraud_ratio=1.0)
    env = ExpertFraudEnv(gen, client)
    env.reset()
    env._true_label = "FRAUD"
    _, reward, terminated, _, info = env.step(1)  # REJECT
    assert terminated
    assert reward == pytest.approx(1.0, abs=1e-6)


def test_gymnasium_compliance():
    """env must pass gymnasium's built-in env checker."""
    from gymnasium.utils.env_checker import check_env
    env = _make_env()
    check_env(env, warn=True)


def test_episode_runs_to_completion():
    env = _make_env()
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 20:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    assert done, "Episode did not terminate within 20 steps"
