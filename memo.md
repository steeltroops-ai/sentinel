# Strategy Memo: KIVE (Knowledge Integrity Verification Engine)

## The Core Thesis
Generative AI optimizes for the most probable, generic path to a correct answer. Human expertise is asymmetric; it is learned through catastrophic failure, idiosyncratic debugging, and nonlinear career decisions. 

To detect an AI-augmented "expert," we must abandon technical trivia. We do not evaluate correctness. We evaluate the presence of context-specific imperfections that an LLM cannot seamlessly fabricate. We attack the constraints of the adversary's pipeline: latency, semantic homogeneity, and behavioral entropy.

By structuring this as a Partially Observable Markov Decision Process (POMDP), we grant the agent a primary lever: the ability to actively induce friction and acquire information through targeted probes. 

## The 9-Dimensional Signal Architecture (Feature Engineering)
We extract 9 explicit, adversarial signals distributed across isolated microservices. 

1. **Temporal Anchoring Violations (TAV)**
   * Every technology has a verifiable birth date. LLMs hallucinate timelines.
   * Claiming 10 years of Kubernetes experience in 2024 is physically impossible.
   * We cross-reference the claimed duration against a pre-computed technology knowledge graph.

2. **Specificity Variance Profile (SVP)**
   * Authentic experts are hyper-specific in their niche and vaguely generalized outside of it.
   * LLMs maintain uniform fluency across all topics. They lack awareness of what they do not know. 
   * We measure variance of named-entity density. Low variance = high AI probability.

3. **Failure Memory Deficiency (FMD)**
   * Real engineers remember catastrophic outages and database locks. 
   * Fraudsters generate textbook, documentation-perfect answers devoid of "scar tissue."
   * We scan sentence constructs for past-tense negation and situational debugging terms.

4. **Market Demand Correlation (MDC)**
   * Sudden retroactive skill inflation correlating tightly with market hype is an anomaly.
   * We detect backdated injections of popular keywords (e.g., "LangChain," "RAG") into prior, unrelated job entries.

5. **Trajectory Smoothness Index (TSI)**
   * Real careers have gaps, horizontal pivots, and tenure volatility.
   * A perfectly monotonic Junior-to-Principal trajectory over exactly five years without gaps is statistically fabricated.

6. **Behavioral Entropy Service (BES)**
   * LLMs rely on copy-pasting or automated transcription software.
   * We capture keystroke telemetry variance, paste-ratios, and window-blur events.
   * Flat, non-bursty keystroke entropy indicates paste-and-send behavior.

7. **Linguistic Quality Assurance (LQA)**
   * Generative models cluster around specific phrasing structures and vocabulary.
   * We compute artifact density, scanning for syntactical tells uniquely produced by autoregressive text generation.

8. **Cross-Candidate Similarity (CCS)**
   * Fraud rings reuse the exact same prompt engineering setups.
   * We project profiles into high-dimensional embedding space and flag tight clusters of identical "expert" personas.

9. **Response Latency Slope (RSL)**
   * Human cognitive load scales non-linearly with problem difficulty. 
   * An LLM pipeline has a fixed, linear token-generation latency, regardless of conceptual density.
   * We detect flattened response latency slopes across highly difficult technical inquiries.

## The Reinforcement Learning Framework
Vetting is a sequential, partial-information game. We train an RL orchestrator to decide when to gather evidence and when to drop the hammer.

* **State Space:** 18D continuous vector encompassing belief, confidence, the 9 microservice scores, normalized probe counts, and live multi-modal data streams perfectly representing the candidate's state.
* **Action Space:** Discrete 8-dimensional space.
   * `PASS` / `REJECT`: Terminal decisions.
   * `FLAG`: Escalation to a human operator when max uncertainty persists.
   * `PROBE`: Targeted interrogations on specific weak-confidence domains.
   * `ESCALATE_DIFFICULTY`, `PROBE_FAILURE_STORY`, `REQUEST_CODE_EXERCISE`, `HOLD`: Active adversarial friction mechanisms.
* **Asymmetric Reward Calculus:** 
   * `PROBE (-0.1)`: Information acquisition is never free.
   * `TRUE PASS / TRUE REJECT (+1.0)`: Correct terminal state.
   * `FALSE POSITIVE (-1.0)`: Rejecting a real expert is an opportunity cost.
   * `FALSE NEGATIVE (-2.5)`: Hiring a fraudster destroys platform credibility. The agent is mathematically punished for unsafe approvals.
* **Agent Core:** We implement `RecurrentPPO` with a `MlpLstmPolicy`. The Recurrent architecture allows the LSTM to retain hidden states across the probe sequences, effectively modeling the candidate's cumulative psychological breakdown over the interview.

## The End-State (Production Paradigm)
This is not a static classifier. It is a live, autonomous interrogator. 

By pushing the vetting process out to multi-modal live telemetry (gaze matching, audio latency tracking, keystroke analysis) and forcing the candidate to defend their knowledge in real-time, we break the attacker's compute pipeline. When an LLM delays an audio response by two seconds to process context, the agent will fire an `INTERRUPT` action, breaking the context window. 

The defense is absolute. It is a mathematical counter-measure to the systemic deterioration of remote hiring trust.
