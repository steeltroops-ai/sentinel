# Multi-Modal Live Evaluator
**Bonus Section | Real-Time Video Interview Fraud Detection**

---

## The Live Interview Problem

Resume fraud is solvable with static signals. Live interview fraud is harder because the fraudster adapts in real-time. They can see your questions, adjust their strategy, and use AI tools during the call if you are not measuring the right behavioral fingerprints.

The solution is not more signals. It is different signals. Behavioral telemetry that reveals AI dependency through patterns the fraudster cannot see or control. Gaze tracking, prosodic variance, and screen interaction events are invisible to the candidate. They cannot game what they do not know you are measuring.

This is not speculative. Eye-tracking hardware is commodity. Screen recording is standard in remote interviews. Audio feature extraction is solved. The infrastructure exists. The question is what to measure and how to act on it in real-time.

---

## Multi-Modal Signal Architecture

**Visual Channel: Gaze Entropy**

Real experts retrieve information from memory. Fraudsters read it from a screen. The eye movement patterns are distinct. Memory retrieval produces high-entropy saccades (random access across visual field). Reading produces low-entropy fixations (left-to-right scan). Pupil dilation spikes during cognitive load for real experts. It stays flat for fraudsters because they are not thinking, they are transcribing.

Signal: `gaze_entropy_score` - Low entropy during technical questions flags reading behavior. High entropy during recall questions confirms genuine expertise.

**Audio Channel: Prosodic Variance**

Human speech modulates. Pitch rises during uncertainty. Tempo slows during complex explanations. Filler words ("um", "uh") cluster before technical terms when experts search for precision. AI-generated responses are monotone because the model does not experience cognitive load. It outputs tokens at constant rate.

Response latency divergence is the strongest audio signal. Real experts answer domain questions faster than adjacent topics because retrieval is automatic. Fraudsters show uniform latency because every answer requires a ChatGPT query. The slope across difficulty tiers is the feature, not any single latency value.

Signal: `prosody_variance_score` - Low variance flags scripted or generated responses. Uniform latency across question types confirms AI dependency.

**Screen Channel: Interaction Anomaly**

Window focus events are binary fraud indicators. Blur to external application during "thinking" pause means they are querying ChatGPT. Paste events during live coding challenges mean they copied a solution. Keystroke dynamics distinguish typing (variable inter-key timing) from pasting (burst insertion).

The fraud sequence has a distinct fingerprint: 8-15 second window blur, refocus, paste or rapid typing burst. This pattern is invisible to the fraudster because they do not know you are logging focus events.

Signal: `interaction_anomaly_score` - Detects context-switching to AI tools through window blur duration and clipboard activity.

---

## Live RL Training Approach

The agent operates in a streaming POMDP. Observations arrive incrementally (video frames at 30fps, audio chunks at 100ms, screen events in real-time). The state vector expands to 24 dimensions: 3 passive multi-modal scores, 4 active probe results, belief state, temporal features (elapsed time, question count, probe history).

The action space extends to include probe types:
- **Continue observing** (accumulate evidence, -0.01/sec latency cost)
- **Probe: Request failure story** (targets FMD signal, -0.05 cost)
- **Probe: Request live coding** (targets BES signal, -0.10 cost)
- **Probe: Ask timeline clarification** (targets TAV signal, -0.05 cost)
- **Escalate to human** (flag for expert review, terminal action)
- **Pass / Reject** (terminal decision)

The reward structure remains asymmetric. False negatives are catastrophic (-2.5). False positives cost revenue (-1.0). Latency penalty (-0.01/sec) incentivizes fast decisions. The agent learns to probe aggressively when uncertainty is high, then commit quickly once confidence threshold is reached.

Training data comes from two sources: simulated interviews with actors following fraud scripts, and real expert recordings labeled retrospectively by client ratings. Fraudster data is augmented by having real experts deliberately use ChatGPT during interviews, capturing ground-truth behavioral shifts when AI assistance is introduced.

---

## Adversarial Robustness

Multi-modal signals are adversarially robust because they require simultaneous deception across independent channels. A fraudster aware of gaze-tracking must fake memory retrieval patterns while maintaining prosodic variance while avoiding window-switching. This is cognitively expensive and creates inconsistencies.

The system's strength is cross-modal correlation. Even if one channel is gamed, others reveal the deception. If gaze entropy is high (faking memory retrieval) but prosody variance is low (reading generated text), the correlation breaks. If window focus is clean but keystroke dynamics show paste events, the correlation breaks.

The fraudster cannot optimize for all channels simultaneously without actually becoming an expert. That is the point.

---

## Production Deployment

**Hardware requirements:** Webcam with 720p+ resolution for gaze tracking. Standard microphone for audio capture. Screen recording via browser API or desktop agent.

**Latency target:** <500ms per observation update. Real-time inference requires edge deployment (client-side feature extraction) with cloud aggregation (server-side belief updates).

**Privacy considerations:** All telemetry is consent-based and disclosed pre-interview. Data retention is session-scoped (deleted after decision). No biometric storage. Compliance with GDPR, CCPA, and biometric privacy laws.

**Fallback strategy:** If hardware fails (no webcam, audio issues), system degrades gracefully to text-only signals (TAV, SVP, FMD). Multi-modal signals are additive, not required.

This is the trust layer for live expert verification. It scales with ProNexus as the platform grows.
