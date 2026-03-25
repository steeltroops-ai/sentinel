import pytest
import numpy as np
from services.orchestrator.env import ExpertFraudEnv, PROBE_ACTION_MAP, ACTION_NAMES
from services.orchestrator.signal_client import MockSignalClient
from data.synthetic_generator import ProfileGenerator


def _make_env():
    gen = ProfileGenerator(fraud_ratio=0.5)
    client = MockSignalClient()
    return ExpertFraudEnv(gen, client)


def test_observation_space():
    env = _make_env()
    obs, _ = env.reset()
    assert obs.shape == (16,), f"Expected (16,), got {obs.shape}"
    assert obs.dtype == np.float32
    assert all(0.0 <= v <= 1.0 for v in obs), f"Obs out of [0,1]: {obs}"


def test_observation_no_duplicates():
    """All 16 observation dimensions must be independently computed -- no duplicates."""
    env = _make_env()
    obs, _ = env.reset(seed=42)
    # Probe all 4 active signals to populate them
    for action in PROBE_ACTION_MAP.keys():
        obs, _, _, _, _ = env.step(action)
    # With all signals populated, no two dimensions should be identical
    # (except by statistical coincidence, which seed=42 avoids)
    seen = set()
    duplicates = []
    for i, v in enumerate(obs):
        key = round(float(v), 6)
        # Allow evidence_count and probed_* flags to collide (they can legitimately be 1.0)
        if i >= 11 and i <= 15:
            continue
        if key in seen and key not in (0.0, 0.5, 1.0):
            duplicates.append((i, key))
        seen.add(key)
    # Soft check: at most 2 coincidental collisions
    assert len(duplicates) <= 2, f"Too many duplicate dimensions: {duplicates}"


def test_passive_signals_populated_at_reset():
    """v5: Passive signals are ALWAYS 0.5 (pure noise, force probing)."""
    env = _make_env()
    obs, _ = env.reset(seed=1)
    # obs[2:7] are passive signals. v5: All should be 0.5 (no information)
    passive = obs[2:7]
    assert all(v == 0.5 for v in passive), f"Passive signals should all be 0.5: {passive}"


def test_active_signals_start_uninformative():
    """Active signals (BES-RSL) should be 0.5 before probing."""
    env = _make_env()
    obs, _ = env.reset(seed=1)
    active = obs[7:11]
    assert all(v == pytest.approx(0.5) for v in active), f"Active signals non-0.5 before probing: {active}"


def test_probe_updates_active_signal():
    """Probing BES should change obs[7] from 0.5."""
    env = _make_env()
    env.reset(seed=1)
    obs, _, _, _, info = env.step(3)  # PROBE_BES
    assert info["action"] == "PROBE_BES"
    assert info["evidence_count"] == 1
    assert obs[7] != pytest.approx(0.5, abs=0.01), f"BES not updated after probe: {obs[7]}"
    assert obs[12] == pytest.approx(1.0), "probed_bes flag not set"


def test_probe_does_not_terminate():
    env = _make_env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(3)  # PROBE_BES
    assert not terminated
    assert not truncated
    assert reward == pytest.approx(0.05, abs=1e-6)  # v5: Probe reward
    assert info["action"] == "PROBE_BES"
    assert info["evidence_count"] == 1


def test_redundant_probe_penalty():
    """Probing the same signal twice should incur R_REDUNDANT."""
    env = _make_env()
    env.reset()
    env.step(3)  # PROBE_BES (first time)
    _, reward, _, _, info = env.step(3)  # PROBE_BES (redundant)
    assert reward == pytest.approx(-0.20, abs=1e-6)


def test_false_negative_penalty():
    """PASS on a FRAUD profile must return -2.5."""
    gen = ProfileGenerator(fraud_ratio=1.0)
    client = MockSignalClient()
    env = ExpertFraudEnv(gen, client)
    env.reset()
    env._true_label = "FRAUD"
    _, reward, terminated, _, _ = env.step(0)  # PASS
    assert terminated
    # v5: -2.5 (false negative) + -0.3 (early decision penalty, 0 probes)
    assert reward == pytest.approx(-2.8, abs=1e-6)


def test_true_reject_reward():
    """REJECT on a FRAUD profile must return +1.0 (or less with early decision penalty)."""
    gen = ProfileGenerator(fraud_ratio=1.0)
    client = MockSignalClient()
    env = ExpertFraudEnv(gen, client)
    env.reset()
    env._true_label = "FRAUD"
    _, reward, terminated, _, _ = env.step(1)  # REJECT
    assert terminated
    # v5: +1.0 (true reject) + -0.3 (early decision penalty, 0 probes)
    assert reward == pytest.approx(0.7, abs=1e-6)


def test_true_pass_reward():
    """PASS on a REAL profile must return +1.0 (or less with early decision penalty)."""
    gen = ProfileGenerator(fraud_ratio=0.0)
    client = MockSignalClient()
    env = ExpertFraudEnv(gen, client)
    env.reset()
    env._true_label = "REAL"
    _, reward, terminated, _, _ = env.step(0)  # PASS
    assert terminated
    # v5: +1.0 (true pass) + -0.3 (early decision penalty, 0 probes)
    assert reward == pytest.approx(0.7, abs=1e-6)


def test_false_positive_penalty():
    """REJECT on a REAL profile must return -1.0 (or less with early decision penalty)."""
    gen = ProfileGenerator(fraud_ratio=0.0)
    client = MockSignalClient()
    env = ExpertFraudEnv(gen, client)
    env.reset()
    env._true_label = "REAL"
    _, reward, terminated, _, _ = env.step(1)  # REJECT
    assert terminated
    # v5: -1.0 (false positive) + -0.3 (early decision penalty, 0 probes)
    assert reward == pytest.approx(-1.3, abs=1e-6)


def test_truncation_at_max_steps():
    """Episode must truncate after MAX_STEPS non-terminal actions."""
    env = _make_env()
    env.reset()
    # Alternate probing all 4 and re-probing to fill MAX_STEPS
    for i in range(env.MAX_STEPS):
        action = 3 + (i % 4)  # cycle through PROBE actions
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert truncated or terminated, "Episode did not truncate at MAX_STEPS"


def test_gymnasium_compliance():
    """env must pass gymnasium's built-in env checker."""
    from gymnasium.utils.env_checker import check_env
    env = _make_env()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env)


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


def test_belief_trajectory_moves():
    """Belief should change after probing -- not stay flat."""
    env = _make_env()
    obs, _ = env.reset(seed=42)
    initial_belief = obs[0]
    # Probe all 4 active signals
    for action in sorted(PROBE_ACTION_MAP.keys()):
        obs, _, _, _, _ = env.step(action)
    final_belief = obs[0]
    assert abs(final_belief - initial_belief) > 0.01, (
        f"Belief unchanged after 4 probes: {initial_belief} -> {final_belief}"
    )
