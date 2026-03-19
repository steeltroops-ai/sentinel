```
INTELLIGENCE OPERATIONS BRIEF
```
# KIVE — Knowledge Integrity Verification Engine

## ProNexus ML Engineer Technical Assignment

```
Complete Intelligence Package: Situation Assessment + System Design + Production Architecture
```
```
Candidate
Mayank Pratap
Singh
```
```
Target Role
ML Engineer #
```
```
Company
ProNexus (NY/SF)
```
```
Classification
HIGH PRIORITY
```
## Section 1 — Situation Intelligence Assessment

##### 1.1 Can You Get This Job?

```
INTELLIGENCE VERDICT
Yes — with non-trivial risk factors. This is a reachable target given the signal alignment
between your background and the role requirements. However, the structural disadvantages
(experience gap, timezone delta, small team risk tolerance) create friction that the
submission quality must neutralize. The assignment is the battlefield. Win there, and the rest
becomes negotiable.
```
**Advantages — Where You Overindex**

- **MediLens exact analog:** Trust scoring from heterogeneous noisy signals is the same problem
    domain. 91-99.8% accuracy across 10 modules. This is not a stretch — it is a direct mapping.
- **Founder-level mindset:** INEXIA VR + 8 hackathon wins + DroidRobotics VP. They want
    someone who treats this like founding responsibility. You have done that literally.
- **End-to-end shipping proof:** FastAPI + PyTorch + AWS + ZenML + MLflow. The MLOps-
    HousePricePrediction repo and deployed projects demonstrate production thinking, not
    academic prototyping.
- **Transformer-from-scratch:** Pytorch-Transformer repo signals deep understanding, not
    framework-level API calling. Rare in the candidate pool for a role at this level.
- **Low-ego builder profile:** INTJ + technical depth + "no hand-holding" in your own cover letter
    matches their stated cultural requirements verbatim.

**Risk Factors — Where You Must Compensate**

- **Experience gap:** They state 2+ years. You have ~14 months of professional SWE (SS
    Innovations, Oct 2024 - Nov 2025). The gap exists. The submission quality and project depth
    must close it.
- **Currently unemployed:** Since Nov 2025. This will surface. The narrative must be "building in
    public" not "searching." OmniContext and MediLens competition participation is the counter.
- **Timezone:** They prefer Eastern Time. India is +9:30 to +10:30 delta. They note "hires remotely
    everywhere" — but early-stage startups run on overlap. This needs to be addressed proactively.


- **No direct RL production experience:** The role explicitly asks for RL. Your exposure is
    academic + project-level. The assignment must demonstrate you think like someone who has
    shipped RL systems, even if the production history is thin.

##### 1.2 Target Analysis — ProNexus

ProNexus is an AI-native expert network platform competing in the primary research / expert sourcing
market currently dominated by GLG and AlphaSights. The differentiation is AI-first: automated sourcing,
vetting, and credibility scoring. They are pre-Series A or early Series A based on the team size and
compensation range. The $130k-$210k band at 3-person team size means external capital exists.

```
Comp range $130k - $210k + 0-2% equity
```
```
Team size 3 people post-hire (you would be #3)
```
```
Stage Pre-Series A / early seed, external capital confirmed by comp range
```
```
Market Expert network / primary research intelligence — GLG, AlphaSights
are incumbents
```
```
Technical moat AI-native vetting — expertise fraud detection is the core defensible IP
```
```
Your leverage ML hire #1. Everything you build becomes foundational.
```

### Section 2 — Job Description Dissection

##### 2.1 What They Said vs. What They Mean

Every JD contains two layers: the stated requirements and the actual evaluation criteria. Reading only
the surface is how candidates lose to those who read the subtext.

```
JD Statement What They Actually Mean Your Signal
```
```
"2+ years ML experience" Can you think in systems and
ship? Years are a proxy for
judgment.
```
```
MediLens 10-module
platform is the counter.
```
```
"Messy real-world problems" We hate people who chase
benchmark scores. We want
engineers who fix production
bugs at 2am.
```
```
Framed correctly in your
cover letter.
```
```
"High ownership" You will be the entire ML team.
There is no one to escalate to.
```
```
INEXIA founder history
proves this.
```
```
"Move from idea to production
quickly"
```
```
We cannot afford a 6-month
R&D cycle. We need
deployable outputs in days to
weeks.
```
```
FastAPI + ZenML + AWS
deployment history.
```
```
"Human-first intuition" If your submission looks like
ChatGPT wrote it, you are
disqualified. They will know.
```
```
The POMDP / Probe
action framing is
distinctly non-generic.
```
```
"Direct feedback early and often" They will fire quickly if
performance is not there. This
is a high-stakes bet.
```
```
You need to outperform
from day one. High risk,
high reward.
```

### Section 3 — Assignment Intelligence Brief

##### 3.1 What They Are Actually Testing

```
EVALUATOR PERSPECTIVE
Erica has read 20+ AI-generated submissions. They all have perplexity scores, burstiness
analysis, and a simple reward function of +1 correct / -1 wrong. She is looking for the ONE
submission that shows a human actually thought about what fraud looks like from the inside
— someone who has seen bad actors and designed around them, not someone who
searched "fraud detection ML" and assembled the top results.
```
##### 3.2 Assignment Parts — Decoded

**Part 1: Strategy Memo (500 words max)**

This is not a summary document. It is a bet. You are being asked to take a single architectural position,
defend it in 500 words, and show that your judgment is better than the alternatives. The evaluator will
read this in 2 minutes and decide if you think differently.

- **Feature Engineering:** Must describe red flags that a human fraudster would exhibit, not just AI-
    generated text. Temporal anchoring violations catch both.
- **RL Framework:** State space, actions, reward function must be explicit and non-generic. The
    Probe action is what differentiates.
- **Ground Truth:** This is the hardest sub-question. Most candidates skip it or give a vague
    answer. ProNexus already has post-call ratings — that is your answer.

**Part 2: Technical Implementation**

Python proof-of-concept. They explicitly say: "We are looking for how you structure the Observation
Space and Penalty or Reward logic." The agent architecture is secondary. The environment design is
primary.

- Build gym.Env with the right observation and action spaces
- Synthetic data generator that makes the differentiation between real and fraud experts visible
- RL agent that learns — show the learning curve in matplotlib
- Episode traces showing probe sequences and decision paths

**Bonus: Multi-Modal Live Evaluator**

Do not skip this. It is "bonus" but it signals whether you can think architecturally beyond the immediate
task. Write this as a design document, not code. 200-300 words with specific signal names and RL action
specification.

**Submission**

Email to erica@pronexus.ai. GitHub repo + memo PDF. Bonus: YouTube walkthrough. The video is
underrated — it is the highest signal of communication ability and genuine understanding.


### Section 4 — KIVE Production System Design

##### 4.1 Architecture Overview

KIVE is designed as a fully decoupled microservices system where every detection signal is an
independent, deployable service with its own API contract. This design ensures that any organization can
adopt individual signal services without requiring the full stack, and that each service can be iterated,
retrained, and scaled independently.

```
Figure 1 — KIVE Microservices Architecture: fully decoupled signal detection pipeline with RL orchestration
```
##### 4.2 Signal Services — Decoupled Detection Modules

Each signal is an independent microservice exposing a REST API. Services are stateless, containerized
(Docker), and can be deployed independently or composed via the RL Orchestrator.

### #

```
TAV
```
###### Temporal Anchoring Violations Service

```
Maintains a technology timeline knowledge graph
(release dates, major versions, adoption inflection
points). Validates claimed experience duration
against physical possibility. Flags when claimed years
> 85% of max possible given release date.
Secondary: checks if expertise claim predates
mainstream adoption (GitHub star inflection point).
This signal is uniquely resistant to AI-assisted fraud
because it operates on externally verifiable
timestamps, not text analysis.
```
```
Weight
```
## 0.

```
NOVEL SIGNAL
```

```
API: POST /api/v1/signals/tav
```
### #

```
SVP
```
###### Specificity Variance Profile Service

```
Computes per-topic specificity scores using NLP
(SpaCy NER density + technical term precision +
numerical concreteness). Calculates variance across
topics. Real experts are non-uniform — hyper-specific
in their domain, openly vague outside it. AI-generated
answers are uniformly fluent. Low SVP variance =
high fraud probability. The insight is to measure
variance, not mean specificity.
API: POST /api/v1/signals/svp
```
```
Weight
```
## 0.

```
NOVEL SIGNAL
```
### #

**FMD**

###### Failure Memory Deficiency Service

```
Scans screening answers for failure narrative
patterns: negation + past tense + specific context +
version references. Real experts remember versions
that broke things, workarounds they had to
implement, libraries they would not use again. AI-
generated text optimizes for correctness. Pattern
matching uses both regex (negation-past constructs)
and semantic similarity to a curated failure narrative
embedding bank.
API: POST /api/v1/signals/fmd
```
```
Weight
```
## 0.

```
NOVEL SIGNAL
```
### #

**MDC**

###### Market Demand Correlation Service

```
Cross-references LinkedIn profile delta timestamps
with Google Trends / job posting frequency data per
skill. Detects retroactive skill inflation: bulk skill
additions 2-4 weeks after market demand spike =
fabrication signal. Requires access to LinkedIn
scraping or profile snapshot history. Most effective as
a secondary signal to corroborate other findings.
API: POST /api/v1/signals/mdc
```
```
Weight
```
## 0.

```
ESTABLISHED
```
### #

```
TSI
```
###### Trajectory Smoothness Index Service

```
Models career progression as a directed graph.
Computes seniority delta between consecutive roles.
Penalizes perfect monotone upward trajectory
(unrealistically smooth). Rewards authentic variance:
lateral moves, gap explanations, down-steps, pivots.
Real careers have bumps. Fabricated ones trend
linearly upward with unrealistically progressive role
labels.
API: POST /api/v1/signals/tsi
```
```
Weight
```
## 0.

```
ESTABLISHED
```

_Figure 2 — Left: Signal weights by detection module. Right: Asymmetric reward matrix (false negatives cost 2.5x false
positives)_


### Section 5 — RL Framework Specification

##### 5.1 Framework: POMDP with Active Probing

The key architectural insight is that expert vetting is not passive classification — it is an interactive
process. The agent can generate targeted follow-up questions to reduce uncertainty. This transforms the
problem from a static bandit into a Partially Observable MDP where the Probe action is the primary lever
for improving decision quality.

**State Space (Observation Vector)**

```
Dimension Range Description
```
**fraud_belief** [0.0, 1.0] (^) Current posterior probability that the expert is
a fraud. Updated via Bayesian update on
each new signal or probe response.
**confidence** [0.0, 1.0] (^) Certainty in the current belief. Low confidence
= high uncertainty = FLAG or PROBE
preferred over terminal actions.
**tav_score** [0.0, 1.0] (^) Temporal anchoring violation severity. 0 =
clean, 1 = severe violation detected.
**svp_variance** [0.0, 1.0] (^) Specificity variance across topics. 0 = high
variance (real expert signal), 1 = uniform
fluency (fraud signal).
**fmd_score** [0.0, 1.0] (^) Failure memory presence. 0 = rich failure
narratives, 1 = no failure memory (fraud
signal).
**evidence_count** [0, 5] (^) Number of probe interactions completed so
far. Controls probe cost accumulation.
**Action Space — Discrete(4)
ID Action Trigger Condition Reward
0** (^) **PASS** fraud_belief < 0.25 AND confidence >
0.
TP: +1.0 / FN:

- 2.

**1** (^) **REJECT** fraud_belief > 0.75 AND confidence >
0.
TN: +1.0 / FP:

- 1.

**2** (^) **FLAG** belief 0.4-0.6 OR confidence < 0.
OR evidence_count >= 5
+0.3 if human
correct
**3** (^) **PROBE(topic)** Uncertainty high AND evidence_count
< 5. Agent selects weakest signal
dimension to probe.

- 0.1 per probe


```
Figure 3 — RL Episode flow: Ingest > Extract > Observe > Decide > (Probe loop max 5) > Terminate
```
##### 5.2 Ground Truth Strategy

```
THE COLD-START ANSWER
ProNexus already has the ground truth if the platform has been running. Three sources: (1)
Post-call client ratings — experts who get low quality ratings are retrospective fraud
positives. (2) Structured probe suite — a bank of technical questions with known correct
answers used pre-placement. (3) Longitudinal score decay — experts with repeated low
post-call scores become positive training examples over time. This is not a cold-start
problem.
```

### Section 6 — Production API Specification

##### 6.1 Unified Signal API Contract

All signal services share a common request/response schema for composability. The RL Orchestrator
calls each service with the same input format and receives a normalized score + explanation +
confidence.

**Request Schema (all signal services)**

```
POST /api/v1/signals/{signal_name}
Authorization: Bearer <api_key>
Content-Type: application/json
```
```
{
"candidate_id": "uuid-v4",
"profile": {
"employment_history": [...], // List of roles with dates, titles, skills
"skill_timestamps": {...}, // Skill: first_claim_date mapping
"education": [...]
},
"screening_responses": [
{"question_id": "q1", "answer": "...", "latency_ms": 4200}
],
"web_signals": {
"github_repos": [...], // Optional: public commit history
"linkedin_delta": [...] // Optional: profile snapshot diff
},
"session_context": {
"prior_probes": [...], // Previous Q&A in session
"evidence_count": 2
}
}
```
**Response Schema (all signal services)**

```
{
"signal": "TAV", // Signal service identifier
"score": 0.78, // [0.0, 1.0] — fraud probability
contribution
"confidence": 0.91, // How certain is this score
"weight": 0.28, // Signal weight in final belief
computation
"flags": [
{
"type": "temporal_violation",
"description": "Claims 6y Kubernetes exp. K8s 1.0 released Jul 2015.
Max=10.7y, claimed=6y — within range but check other signals.",
"severity": "low",
"evidence": {"tool": "kubernetes", "claimed_years": 6, "max_possible":
10.7}
}
],
"probe_suggestion": {
"question": "Walk me through a specific Kubernetes networking issue you
debugged in production.",
"target_dimension": "TAV",
"expected_fraud_response_pattern": "Generic answer about pods and services
without version-specific details."
```

```
},
"latency_ms": 143,
"model_version": "tav-v1.2.0"
}
```
##### 6.2 RL Orchestrator API

```
# Start a vetting session
POST /api/v1/orchestrator/session/start
```
```
# Submit probe response (called after candidate answers probe question)
POST /api/v1/orchestrator/session/{session_id}/probe_response
```
```
# Get current decision
GET /api/v1/orchestrator/session/{session_id}/decision
```
```
# Response: { action: "PROBE"|"PASS"|"REJECT"|"FLAG", belief: 0.72,
# confidence: 0.88, probe_question: "...", session_id: "..." }
```
```
# Submit ground truth (post-call rating)
POST /api/v1/orchestrator/ground_truth
# Body: { candidate_id, actual_label: "REAL"|"FRAUD", client_rating: 4.1 }
```
##### 6.3 Full Technology Stack

```
Layer Technology Rationale
```
```
RL Framework Stable-Baselines3 /
Gymnasium
```
```
Standard gym.Env interface, DQN or
PPO agent, proven for discrete
action spaces
```
```
Signal NLP SpaCy + HuggingFace
Transformers
```
```
NER for entity extraction, sentence-
transformers for semantic similarity
(FMD)
```
```
API Layer FastAPI + Pydantic v2^ Async, type-safe, auto-docs,
production-grade from day one
```
```
Tech Timeline KB SQLite + SQLAlchemy^ Technology release date graph —
lightweight, no infra dependency
```
```
ML Experiment Tracking MLflow^ Episode logging, reward curves,
model versioning
```
```
Pipeline Orchestration ZenML^ Signal pipeline as ZenML steps —
reproducible, observable
```
```
Containerization Docker + Docker Compose^ Each signal service in its own
container — true decoupling
```
```
Synthetic Data Faker + custom
generators
```
```
Realistic expert profiles for training
and evaluation
```
```
Visualization Matplotlib + Seaborn^ Learning curves, reward
distributions, episode traces
```
```
Language Python 3.11+^ Ecosystem fit — all ML/NLP
libraries, FastAPI, gymnasium
```


### Section 7 — Strategy Memo Draft

```
NOTE
This is the Part 1 deliverable — the 500-word thesis. Write this in your own voice. Use this as
a structural template. The specific signal names and RL framing are the core differentiation.
Do not soften the positions.
```
#### KIVE — Architectural Thesis

```
Knowledge Integrity Verification Engine | Strategy Memo
```
**The Problem with Current Approaches**

The canonical failure of expertise fraud detection is treating it as a static binary classification problem
over text signals. Perplexity scores and burstiness analysis detect AI-generated text — they do not detect
AI-assisted human fraud, where a real person uses AI to craft responses that sound authentic. In 2025,
this is the dominant attack vector.

My architectural thesis: expertise fraud detection is a Partially Observable MDP, not a classification task.
The critical insight is that a vetting system can ask questions. An agent that actively generates targeted
follow-up questions to reduce uncertainty is fundamentally stronger than one that passively scores what
it was given.

**Feature Engineering — Three Signals That Matter**

**Temporal Anchoring Violations (TAV, weight 0.28):** Every technology has a birth date. Claimed
experience that exceeds what is physically possible given release history is a verifiable lie, independent
of text quality. A fraudster who claim-inflates their Kubernetes experience can write perfectly human-
sounding answers — but if they claim six years when the tool is eight years old and they are twenty-six,
the math narrows the search space dramatically. This signal operates on externally verifiable timestamps,
making it adversarially robust in a way that perplexity analysis is not.

**Specificity Variance Profile (SVP, weight 0.24):** Real experts are non-uniform. They are hyper-specific
in their domain and openly vague outside it. AI-generated responses are uniformly fluent everywhere —
the model does not know what it does not know. Measuring the variance of specificity scores across
topics, not the mean, reveals this. Low variance = high fraud probability.

**Failure Memory Deficiency (FMD, weight 0.20):** Real experts remember what broke. They reference
specific library versions that had bugs, workarounds they implemented, things they would do differently.
AI-generated answers optimize for correctness — they describe best practices, not war stories. Scanning
for failure narrative patterns (negation + past tense + specific context) identifies authentic operational
experience.

**The RL Framework**

State space: six-dimensional vector — fraud_belief [0,1], confidence [0,1], tav_score, svp_variance,
fmd_score, evidence_count. Actions: PASS, REJECT, FLAG, or PROBE(topic). The Probe action is
novel: the agent selects the signal dimension with highest uncertainty and generates a targeted question
to reduce it, at a cost of -0.1 reward per probe.


Reward function is asymmetric by design: +1.0 for correct PASS or REJECT, -2.5 for false negatives (a
hired fraudster causes downstream client damage and platform reputation harm), -1.0 for false positives
(a real expert incorrectly rejected), +0.3 when FLAG leads to a correct human decision. This asymmetry
encodes the actual cost structure of the business.

Ground truth: ProNexus already has it. Post-call client ratings retrospectively label experts. A structured
probe suite (known-answer technical questions) provides pre-placement ground truth. Longitudinal score
decay identifies repeat poor performers as retrospective positives. This is not a cold-start problem.


### Section 8 — 6 - Day Execution Plan

```
Day Phase Deliverable Priority
```
###### 1 Strategy Memo^ Write + finalize the 500-word thesis.

```
Opinionated. No hedging.
TAV/SVP/FMD as the bet. RL as
POMDP. Probe action defined.
```
```
CRITICAL
```
###### 2 Synthetic Data^ expert_generator.py —^ creates

```
realistic real vs fraud profiles with all
signal signatures. This is the training
foundation.
```
```
CRITICAL
```
###### 3 gym.Env^ ExpertFraudEnv: observation space,

```
action space, step logic, probe loop,
reward function. All 5 signals in
observation.
```
```
CRITICAL
```
###### 4 Agent + Training^ DQN or Q-table agent. Train 2k-5k

```
episodes. Learning curve plot. Episode
trace showing probe sequences.
```
```
HIGH
```
###### 5 Bonus + Polish^ Multi-modal design doc (200-^300

```
words). Code cleanup. README.
Repo structure. All tests green.
```
```
HIGH
```
###### 6 Video + Submit^ Loom/YouTube walkthrough: signal

```
taxonomy rationale, environment
demo, learning curve, multi-modal
extension. Email to
erica@pronexus.ai.
```
```
HIGH
```
### Section 9 — Bonus: Multi-Modal Live Evaluator Design

##### 9.1 Signal Extraction in Live Video Interviews

The live setting transforms the observation space from static profile signals to a continuous multi-modal
stream. Three channels run simultaneously.

```
Channel Cue / Feature Fraud Signature
```
```
Visual Gaze deviation on claim-
heavy statements vs
general answers
```
```
Consistent upward-left gaze when stating
specific numbers/dates = confabulation
pattern
```
```
Visual Micro-expression latency
before answering
```
```
Real expert: immediate answer or honest
"I need to think." Fraud: uniform ~2.5s
delay (AI query time)
```
```
Audio Prosody analysis: pitch
variance on known-wrong
answers
```
```
Detecting overconfidence phoneme
patterns — flat affect on technically
incorrect claims
```

```
Audio Response latency by
question difficulty tier
```
```
Hard questions should take longer for
real experts. Uniform latency across
difficulty = AI-assisted.
```
```
Screen Tab switch detection during
screenshare
```
```
Alt-Tab to external resource on technical
questions is a direct fraud signal
```
```
Screen Keystroke timing patterns
during typing exercises
```
```
Copy-paste timing signature vs. real-time
typing on code exercises
```
##### 9.2 Live RL Action: Real-Time Difficulty Escalation

In the live setting, the PROBE action evolves into a difficulty escalation mechanism. The agent observes
the multi-modal signal stream in real time and selects from an extended action space: ESCALATE
(increase question difficulty tier), PROBE_DEEP (ask for a specific failure story), HOLD (continue at
current difficulty), TERMINATE_PASS, TERMINATE_REJECT, or FLAG_HUMAN.

Training requires labeled session recordings annotated with post-call verification outcomes. The agent
learns to escalate difficulty when multi-modal signals are ambiguous, and to terminate quickly when
signals are highly confident in either direction. The asymmetric reward structure carries over from the
static setting.

### Section 10 — Repository Structure

```
kive/
├── README.md # Full project overview + running instructions
├── memo.pdf # Part 1: Strategy memo
├── requirements.txt
├── docker-compose.yml # All services composable
├── services/
│ ├── tav/ # Temporal Anchoring Violations service
│ │ ├── Dockerfile
│ │ ├── main.py # FastAPI app
│ │ ├── detector.py # TAV logic + tech timeline KB
│ │ └── tech_timeline.db # SQLite technology release DB
│ ├── svp/ # Specificity Variance Profile service
│ ├── fmd/ # Failure Memory Deficiency service
│ ├── mdc/ # Market Demand Correlation service
│ ├── tsi/ # Trajectory Smoothness Index service
│ └── orchestrator/ # RL Orchestrator + Probe Generator
│ ├── env.py # ExpertFraudEnv (gym.Env)
│ ├── agent.py # DQN agent
│ ├── train.py # Training loop + learning curves
│ └── main.py # FastAPI orchestrator API
├── data/
│ ├── synthetic_generator.py # Fake expert profile generator
│ └── probe_suite/ # Known-answer technical question bank
├── notebooks/
│ ├── 01_signal_analysis.ipynb
│ └── 02_rl_training.ipynb # Episode traces, learning curves
└── tests/
├── test_tav.py
├── test_env.py
└── test_integration.py
```


```
END OF INTELLIGENCE BRIEF
```
### KIVE — Knowledge Integrity Verification Engine

_Prepared by Mayank Pratap Singh | ProNexus ML Engineer Assignment | March 2026_

```
github.com/steeltroops-ai | steeltroops.vercel.app
```

