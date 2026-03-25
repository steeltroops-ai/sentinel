# Sentinel | KIVE: Knowledge Integrity Verification Engine
**Strategy Memo | Mayank Pratap Singh**

---

## The Actual Problem

There are two of you reading this. You need a third person who owns the intelligence layer completely without being managed into it.

Perplexity scores and burstiness metrics catch lazy fraud. They miss the candidate who spends 20 minutes prompting GPT to sound like themselves, edits the output, and submits something that clears every classifier you have. That is the majority of fraud on this platform and it is accelerating.

My thesis: generative AI optimizes for the most probable correct answer. Human expertise is built from the opposite, through version-specific failures, nonlinear career decisions, and production incidents you remember because they cost you sleep. The signal is not whether the answer is correct. It is whether the answer contains the specific inconvenient context that only comes from having actually done the work.

---

## Feature Engineering

**The five signals I would prioritize, in order:**

| Signal | Weight | Red Flag |
|--------|--------|----------|
| **BES** Behavioral Entropy Service | 0.18 | Keystroke timing variance, paste ratios, window blur duration. The AI query sequence has a distinct fingerprint. Strongest active signal. |
| **TAV** Temporal Anchoring Violations | 0.14 | Claimed experience exceeding physical maximum given tech release dates. Kubernetes (2014), LangChain (2022). Math problem, not inference. |
| **LQA** Linguistic Quality Assurance | 0.12 | Detection of non-latent markers, uniform fluency, and robotic syntax patterns via active stochastic probing. |
| **SVP** Specificity Variance Profile | 0.11 | Experts are hyper-specific in-domain and vague outside. Low cross-topic variance is the fraud signal, not low quality. |
| **FMD** Failure Memory Deficiency | 0.11 | GPT optimizes for correctness. Real engineers have scar tissue: version-specific bugs and decisions they regret. |

Four supporting services (MDC, TSI, CCS, RSL) handle market demand correlation, trajectory smoothness index, cross-candidate similarity, and response latency slope. They feed the same observation vector as secondary weight.

The strongest signals are adversarially robust because fraudsters cannot game what they don't know exists. BES and RSL operate on behavioral fingerprints that are invisible until measured. Each signal is an independent REST service. Any organization can adopt TAV or FMD without the full stack. This is how you build a moat that compounds.

---

## The RL Framework

Vetting is a sequential partial-information game. The agent's advantage over a static classifier is that it can generate targeted follow-up questions before committing to a terminal decision.

**State space (16 dimensions, normalized to [0,1]):** fraud belief and confidence, five passive signal scores, four active probe scores initialized at 0.5, evidence count normalized to probe budget, binary probe execution flags preventing redundant queries.

**Action space: Discrete(7):** PASS, REJECT, FLAG, PROBE\_BES, PROBE\_LQA, PROBE\_CCS, PROBE\_RSL.

| Outcome | Reward | Reasoning |
|---------|--------|-----------|
| True Pass / True Reject | +1.0 | Correct terminal decision |
| False Negative | -2.5 | Fraudster placed destroys platform credibility |
| False Positive | -1.0 | Real expert rejected is direct revenue loss |
| Flag Hit / Flag Miss | +0.3 / -0.2 | Models realistic human operator accuracy |
| Probe | +0.05 | Rewards evidence acquisition (information has value) |
| Redundant Probe | -0.20 | Catastrophic penalty for querying an already-scored signal |
| Early Decision | -0.30 | Penalty for deciding with insufficient evidence (< 2 probes) |

**Agent:** RecurrentPPO with MlpLstmPolicy. The LSTM is structurally necessary because what the agent probed in step 2 changes how it interprets step 4. A feedforward policy loses that sequential dependency entirely.

**Ground truth:** ProNexus already has it. Post-call client ratings retrospectively label experts. A structured bank of known-answer technical questions provides pre-placement verification. Longitudinal low ratings across repeat engagements identify retrospective fraud positives over time. This is not a cold-start problem if the platform has any run history.

The MockSignalClient is calibrated so one-step greedy classification is mathematically suboptimal. Passive signals provide zero class information (always return 0.5). Active signals are highly discriminative but only accessible via probe actions. The agent must probe to acquire any useful information before making a terminal decision.

The multimodal live evaluator extension covering gaze deviation, prosody confidence variance, and screen focus loss sequences is addressed fully in the bonus section with an expanded live observation space.

---

## Why I Am the Third Engineer

BES and RSL are in my observation space as scored microservices with API contracts. Every other submission you receive this week describes them in a paragraph. That difference is the difference between thinking about the problem and building the system.

I built this architecture in a different domain. MediLens combines trust signals across 10 diagnostic modules at 91 to 99.8% production accuracy. The domain changes. The structure does not.

I see where ProNexus is going. You're building the trust layer for the entire expert network industry. I'm here to own that intelligence layer and scale it with you.