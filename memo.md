# Strategy Memo — KIVE (Knowledge Integrity Verification Engine)
# Mayank Pratap Singh | steeltroops.ai@gmail.com

## The Problem

Current expertise fraud detection fails because it targets the wrong threat model.
AI-generated text detectors measure perplexity — a property of the *output*, not
the *operator*. The dominant attack vector is not pure AI generation: it is AI-assisted
human fraud, where a person with surface-level knowledge uses LLMs to produce fluent,
coherent professional responses. Perplexity scores collapse under this scenario because
the human steers the generation and reviews the output. The result reads like a person.
It scores like a person. It is not a person with the claimed experience.

## The Correct Frame

This is a Partially Observable MDP, not a classification task. The vetting agent begins
with high uncertainty about a candidate's true expertise level and must take sequential
actions — collecting signals, asking probe questions — to reduce that uncertainty before
making a costly terminal decision.

The asymmetric cost structure is critical to design for explicitly. A False Negative —
passing a fraudster — costs the platform a damaged client relationship, failed project,
and potential legal exposure. A False Positive — rejecting a real expert — costs one
placement opportunity. Based on the assignment brief, False Negatives carry
approximately 2.5x the business cost of False Positives. This ratio must be encoded
in the reward function, not treated as a post-hoc threshold parameter.

## What KIVE Measures

Five signals, selected for adversarial robustness:

**TAV (Temporal Anchoring Violations, weight 0.28)**: Technology release dates are
external, immutable ground truth. Claimed experience duration that exceeds the
physical maximum since a tool's release date cannot be manufactured with better
prompts. This signal is the most adversarially durable in the system.

**SVP (Specificity Variance Profile, weight 0.24)**: Real experts are hyper-specific
in their domain and openly vague outside it. LLMs produce uniform fluency across all
topics because the model does not have a domain boundary. Measuring variance of
specificity across topics — not mean — reveals the AI assistance pattern.

**FMD (Failure Memory Deficiency, weight 0.20)**: LLMs optimize for correctness. They
do not naturally produce specific production failure narratives with version numbers,
root causes, and personal accountability. Real engineers have scars. Absence of war
stories across multiple prompts is a strong, durable fraud indicator.

**MDC (Market Demand Correlation, weight 0.16)**: Retroactive skill inflation — 
bulk skill additions 0-3 months after demand spikes — indicates resume padding to
match job market trends rather than citing organic skill acquisition.

**TSI (Trajectory Smoothness Index, weight 0.12)**: Fabricated CVs trend monotonically
upward. Real careers include lateral moves, gaps, and pivots. A perfectly smooth
progression over 8+ years is statistically anomalous for authentic professional history.

## Why RL is the Right Tool

A static classifier assumes the signal distribution is fixed. It is not. Fraudster
behavior adapts. The probe loop — where the agent actively generates targeted questions
to resolve uncertainty — is only possible with an agent that has learned when
uncertainty is high enough to warrant the information acquisition cost. DQN over a
6-dimensional observation space learns this boundary from episode experience.

The reward function encodes: True Pass = +1.0, True Reject = +1.0, False Negative = -2.5,
False Positive = -1.0, Probe = -0.1. The agent learns to probe when belief is ambiguous
and terminate when confidence justifies the decision cost.

## What This Does Not Claim

KIVE does not claim 100% detection. It claims asymmetric cost alignment and
adversarial durability beyond perplexity-based approaches. The system degrades
gracefully — FLAG routes to human review when the agent lacks sufficient confidence.
Human reviewers receive the full signal breakdown and recommended probe questions,
making their judgment more targeted than an unaided interview.
