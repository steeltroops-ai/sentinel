# KIVE System Architecture

## 1. High-Level Pipeline

```mermaid
graph TD
    subgraph Data_Layer ["Data Generation & Validation"]
        SG["synthetic_generator.py<br>ProfileGenerator"] --> SP[("Synthetic Profiles<br>JSON")]
        VD["validate_distribution.py"] -.->|Verifies Conditional Distributions| SP
    end

    subgraph Signal_Services ["Unified Grid Microservices"]
        TAV["TAV :8001<br>Temporal Anchoring"]
        SVP["SVP :8002<br>Specificity Variance"]
        FMD["FMD :8003<br>Failure Memory"]
        MDC["MDC :8004<br>Market Demand"]
        TSI["TSI :8005<br>Trajectory Smoothness"]
        BES["BES :8006<br>Behavioral Entropy"]
        LQA["LQA :8007<br>Linguistic Quality"]
        CCS["CCS :8008<br>Cross-Candidate Sim"]
        RSL["RSL :8009<br>Response Latency"]
    end

    subgraph Orchestrator ["RL Orchestrator"]
        ENV["ExpertFraudEnv<br>gymnasium.Env"]
        SC["MockSignalClient<br>Local Training Bypass"]
        AGENT["RecurrentPPO<br>MlpLstmPolicy"]
        TRAIN["train.py<br>PPO Loop"]
    end

    subgraph Outputs ["Convergence Artifacts"]
        LC["learning_curve.png"]
        ET["episode_traces.png"]
        CR["convergence_report.json"]
        MODEL["ppo_policy.zip"]
    end

    SP --> ENV
    ENV <--> SC
    SC -.->|API Simulation| Signal_Services
    AGENT <--> ENV
    AGENT --> TRAIN
    TRAIN --> LC
    TRAIN --> ET
    TRAIN --> CR
    TRAIN --> MODEL

    classDef core fill:#1a1f2e,stroke:#4C9BE8,stroke-width:2px,color:#e0e0e0;
    classDef signal fill:#2d1f1f,stroke:#E84C4C,stroke-width:2px,color:#e0e0e0;
    classDef data fill:#1f2d1f,stroke:#4CE88C,stroke-width:2px,color:#e0e0e0;
    classDef output fill:#2d2d1f,stroke:#E8C84C,stroke-width:2px,color:#e0e0e0;
    class ENV,AGENT,TRAIN,SC core;
    class TAV,SVP,FMD,MDC,TSI,BES,LQA,CCS,RSL signal;
    class SG,VD,SP data;
    class LC,ET,CR,MODEL output;
```

---

## 2. Signal Detection Matrix

The core architecture runs on 9 distinct microservices handling isolated vector queries. Each returns a normalized fraud probability score `[0, 1]` coupled with internal confidence logic.

| Signal | Execution | Weight | Core Focus |
|--------|-----------|--------|------------|
| **TAV** | Passive | 0.28   | Resume temporal inconsistencies |
| **SVP** | Passive | 0.24   | LLM linguistic uniformity |
| **FMD** | Passive | 0.20   | Lack of specific failure experiences |
| **MDC** | Passive | 0.16   | Retroactive skill inflation aligning to market trends |
| **TSI** | Passive | 0.12   | Resume monotonicity anomalies |
| **BES** | Active  | 0.18   | Keystroke entropy, UI event telemetry |
| **LQA** | Active  | 0.10   | Token sampling hedging artifacts |
| **CCS** | Active  | 0.08   | Cross-session payload overlaps |
| **RSL** | Active  | 0.07   | Standard latency divergence curves |

---

## 3. POMDP State Transitions

```mermaid
sequenceDiagram
    participant PG as ProfileGenerator
    participant ENV as ExpertFraudEnv
    participant SC as MockSignalClient
    participant AGENT as RecurrentPPO

    PG->>ENV: extract_profile() -> profile, Boolean Class
    ENV->>SC: fetch_passive(profile)
    SC-->>ENV: 5 signal scores (TAV, SVP, FMD, MDC, TSI)
    ENV->>ENV: Construct initial 18D state vector
    ENV->>AGENT: Observation (belief, signals)

    loop Action Sequence (Maximum=10 Steps)
        AGENT->>ENV: invoke_action(PROBE_LQA)
        ENV->>SC: invoke_active(profile, LQA)
        SC-->>ENV: resolved active score
        ENV->>ENV: Compute Bayesian Posterior Shift
        ENV->>AGENT: Observation, Partial Reward (-0.02)
    end

    AGENT->>ENV: invoke_terminal(PASS / REJECT / FLAG)
    ENV->>ENV: Map action validity against True Class
    ENV-->>AGENT: Final Matrix Reward, step_truncation=True
```

---

## 4. Observation State Dimensions

The POMDP represents the environment as an 18-dimensional multimodal continuous vector space. All values are scaled within `[0, 1]`.

| Matrix Vector | Attribute | Limits | Description |
|---|---|---|---|
| `[0]` | `fraud_belief` | `[0, 1]` | Bayesian posterior aggregation of all resolved signals. |
| `[1]` | `confidence` | `[0, 1]` | Relative density and variance agreement of the collected inputs. |
| `[2:6]` | `Passive Base` | `[0, 1]` | `[TAV, SVP, FMD, MDC, TSI]` - Populated upon environment reset condition. Standard initialization 0.5 where MNAR. |
| `[7:10]` | `Active Base` | `[0, 1]` | `[BES, LQA, CCS, RSL]` - Initialized to 0.5. Updated purely upon execution of discrete PROBE actions. |
| `[11]` | `Normalized Step` | `[0, 1]` | Ratio of elapsed discrete cycles versus `MAX_STEPS`. Prevents infinite recurrent loops. |
| `[12:15]` | `Binary Probes` | `{0, 1}` | Indication matrix signaling if `[PROBE_BES, PROBE_LQA, PROBE_CCS, PROBE_RSL]` have fired. |
| `[16]` | `Passive Belief` | `[0, 1]` | Discrete Bayesian weight of the 5 passive components alone. |
| `[17]` | `Active Belief` | `[0, 1]` | Discrete Bayesian weight of the probed components alone. |

---

## 5. Discrete Action Configuration

The action space consists of 7 explicit channels balancing final validation decisions against recursive data acquisition.

**Action Mapping `Discrete(7)`**:
- `0`: PASS
- `1`: REJECT
- `2`: FLAG
- `3`: PROBE_BES (Yields precise behavioral data)
- `4`: PROBE_LQA (Yields localized text analytics)
- `5`: PROBE_CCS (Cross-evaluates past instances)
- `6`: PROBE_RSL (Fetches high-confidence timing metrics)

**Cost Asymmetry**:
- True Positive / True Negative = `+1.0`
- False Negative (Fraud admitted) = `-2.5`
- False Positive (Expert denied) = `-1.0`
- Safe Default (Flagged) = `+0.3` (Hit), `-0.2` (Miss)
- Singular Probe execution = `-0.02`
- Redundant Probe Execution = `-0.20`

---

## 6. Docker Container Infrastructure

The services leverage a single logical root map utilizing YAML component inheritance to achieve zero-redundancy scaling.

```mermaid
graph TD
    subgraph Kive_Internal_Network ["Isolated Engine Runtime"]
        TAV_C["kive-tav :8001"]
        SVP_C["kive-svp :8002"]
        FMD_C["kive-fmd :8003"]
        MDC_C["kive-mdc :8004"]
        TSI_C["kive-tsi :8005"]
        BES_C["kive-bes :8006"]
        LQA_C["kive-lqa :8007"]
        CCS_C["kive-ccs :8008"]
        RSL_C["kive-rsl :8009"]
        ORC_C["kive-orchestrator :8010"]
    end

    ORC_C -->|REST Protocol| TAV_C
    ORC_C -->|REST Protocol| SVP_C
    ORC_C -->|REST Protocol| FMD_C
    ORC_C -->|REST Protocol| MDC_C
    ORC_C -->|REST Protocol| TSI_C
    ORC_C -->|REST Protocol| BES_C
    ORC_C -->|REST Protocol| LQA_C
    ORC_C -->|REST Protocol| CCS_C
    ORC_C -->|REST Protocol| RSL_C

    EXT["External Port Binding<br>Client Application"] --> ORC_C
```

All 10 instances derive from one source `Dockerfile` referencing the shared `requirements.txt`. The Orchestrator resolves endpoints inherently via internal DNS mappings defined strictly in `docker-compose.yml` environment configurations, entirely negating local port conflicts.

---

## 7. Operational Roadmap Expansion

- **State Distribution Normalization:** Introduce Layer Normalization across the Bayesian observation components to stabilize PPO value calculations further.
- **Continuous Action Migration:** Extend Action outputs from isolated discrete selections to continuous multi-probe arrays resolving confidence parameters intrinsically within a single iteration.
- **Real-Time Database Persistence:** Migrate the raw session logging within the Orchestrator `main.py` state dictionaries to a permanent scalable vector backend store (e.g. pgvector, Chroma) for post-mortem analysis and retraining cycles.
- **Dynamic Action Masking:** Completely eliminate the possibility of selecting `PROBE_BES` when binary probe indicator `[12]` is active, enforcing 100% legal actions logically at the agent probability level.
