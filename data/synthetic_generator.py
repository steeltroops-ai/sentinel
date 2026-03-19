# data/synthetic_generator.py
# Generates realistic real vs. fraud expert profiles for KIVE training.
# Fraud patterns: uniform specificity, no failure memory, smooth career, possible TAV violations.
# Real patterns: non-uniform specificity, failure stories, lateral career moves, coherent timestamps.

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import uuid
from typing import Optional

from faker import Faker

fake = Faker()
random.seed(None)


# ─── Taxonomy ────────────────────────────────────────────────────────────────

SENIORITY_MAP = {
    "intern": 0, "trainee": 0,
    "junior": 1, "associate": 1,
    "engineer": 2, "developer": 2, "analyst": 2,
    "senior": 3,
    "staff": 4, "lead": 4,
    "principal": 5, "architect": 5,
    "director": 6, "manager": 5, "head": 6,
    "vp": 7, "vice president": 7,
    "cto": 8, "ceo": 8, "founder": 7, "co-founder": 7,
}

TECH_TIMELINE = {
    "kubernetes": {"release_year": 2014, "inflection_year": 2017},
    "docker":     {"release_year": 2013, "inflection_year": 2015},
    "pytorch":    {"release_year": 2016, "inflection_year": 2018},
    "tensorflow": {"release_year": 2015, "inflection_year": 2017},
    "react":      {"release_year": 2013, "inflection_year": 2015},
    "fastapi":    {"release_year": 2018, "inflection_year": 2020},
    "rust":       {"release_year": 2015, "inflection_year": 2020},
    "langchain":  {"release_year": 2022, "inflection_year": 2023},
    "nextjs":     {"release_year": 2016, "inflection_year": 2019},
    "airflow":    {"release_year": 2014, "inflection_year": 2018},
}

FAILURE_STORY_BANK = [
    {
        "topic": "kubernetes",
        "answer": (
            "We hit a nasty issue on K8s 1.18 — liveness probes were misconfigured "
            "on a StatefulSet, took down 3 replicas simultaneously during a rolling update. "
            "Took 6 hours to trace. Now I always set initialDelaySeconds to 2x startup time "
            "and never allow maxUnavailable > 0 for StatefulSets."
        ),
        "latency_range": (8000, 25000),
    },
    {
        "topic": "pandas",
        "answer": (
            "Pandas 0.24 had a behavior where merge() silently dropped rows with NaN in "
            "the join key. Cost us a production join missing 12% of records for 3 days. "
            "Switched to explicit fillna() + indicator=True to catch drops immediately."
        ),
        "latency_range": (5000, 15000),
    },
    {
        "topic": "pytorch",
        "answer": (
            "Had a GPU memory leak in PyTorch 1.9 caused by not detaching tensors in a "
            "custom backward pass. Ran fine for 50 epochs then OOM. Took 2 days to find — "
            "now always profile memory with torch.cuda.memory_allocated() during early epochs."
        ),
        "latency_range": (7000, 20000),
    },
    {
        "topic": "redis",
        "answer": (
            "Misconfigured eviction policy as allkeys-lru on a Redis instance storing both "
            "session tokens and rate-limit counters. Under load, session tokens got evicted, "
            "logging out active users. Fixed by using two Redis clusters with different policies."
        ),
        "latency_range": (6000, 18000),
    },
    {
        "topic": "general",
        "answer": (
            "Worst production failure I caused was a migration that ran without a dry-run "
            "in production. Wiped 3 months of soft-deleted records that were still being "
            "referenced. Took 14 hours to restore from backups. Now every migration runs "
            "against a prod clone first, always."
        ),
        "latency_range": (10000, 30000),
    },
]

FRAUD_GENERIC_ANSWERS = {
    "kubernetes": (
        "Kubernetes is a container orchestration platform that manages deployment, scaling, "
        "and operations of containerized applications. I use kubectl to manage deployments "
        "and configure resource limits and health checks for production reliability."
    ),
    "pandas": (
        "Pandas is a powerful data manipulation library. I use groupby operations, "
        "merge for joining datasets, and apply with custom functions. I focus on "
        "memory efficiency with large datasets."
    ),
    "pytorch": (
        "PyTorch is my primary deep learning framework. I build neural networks using "
        "nn.Module, use DataLoader for batching, and optimize with Adam. I'm comfortable "
        "with both training from scratch and fine-tuning pretrained models."
    ),
    "general_adjacent": (
        "I have some familiarity with that area and understand the core concepts. "
        "I can certainly learn more about it quickly and apply my existing software "
        "engineering skills to become productive."
    ),
}

REAL_ADJACENT_VAGUE = [
    "Honestly not my primary area — I know the concepts but I'd rely on someone with "
    "deeper experience here. I wouldn't trust myself to architect this without studying more.",
    "I've touched this lightly but I'd call myself a beginner. Happy to learn but "
    "I wouldn't want to overstate what I know.",
    "That's outside my main domain. I know enough to have a conversation but not "
    "enough to be the expert in the room on it.",
]


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Role:
    title: str
    company: str
    start_year: int
    end_year: Optional[int]
    skills: list[str]
    claimed_skills_added_date: dict[str, str]

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ScreeningResponse:
    question_id: str
    answer: str
    latency_ms: int
    topic: str
    question_difficulty: str

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ExpertProfile:
    label: str
    id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    employment_history: list[Role] = dataclasses.field(default_factory=list)
    skill_timestamps: dict[str, str] = dataclasses.field(default_factory=dict)
    screening_responses: list[ScreeningResponse] = dataclasses.field(default_factory=list)
    github_repos: list[dict] = dataclasses.field(default_factory=list)
    linkedin_delta: list[dict] = dataclasses.field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "employment_history": [r.to_dict() for r in self.employment_history],
            "skill_timestamps": self.skill_timestamps,
            "screening_responses": [r.to_dict() for r in self.screening_responses],
            "github_repos": self.github_repos,
            "linkedin_delta": self.linkedin_delta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExpertProfile:
        p = cls(label=d["label"], id=d["id"])
        p.employment_history = [Role(**r) for r in d.get("employment_history", [])]
        p.skill_timestamps = d.get("skill_timestamps", {})
        p.screening_responses = [ScreeningResponse(**r) for r in d.get("screening_responses", [])]
        p.github_repos = d.get("github_repos", [])
        p.linkedin_delta = d.get("linkedin_delta", [])
        return p


# ─── Generators ──────────────────────────────────────────────────────────────

class RealExpertGenerator:

    def generate(self) -> ExpertProfile:
        profile = ExpertProfile(label="REAL")
        career_start = random.randint(2012, 2020)
        primary_domain = random.choice(["ml", "backend", "devops", "data"])

        profile.employment_history = self._build_career(career_start)
        profile.skill_timestamps = self._build_timestamps(career_start, primary_domain)
        profile.screening_responses = self._build_responses(primary_domain)
        return profile

    def _build_career(self, start_year: int) -> list[Role]:
        TITLE_SEQUENCES = [
            # Perfect upward (legitimate, but rarer)
            ["Junior Engineer", "Engineer", "Senior Engineer", "Staff Engineer"],
            # With lateral move (same level, different domain)
            ["Engineer", "Senior Engineer", "Senior Engineer", "Staff Engineer"],
            # Down-step then recovery (startup bet or deliberate pivot)
            ["Senior Engineer", "Engineer", "Senior Engineer", "Principal Engineer"],
            # Gap + pivot (left to do something else)
            ["Junior Engineer", "Engineer", "Senior Engineer"],
            # Startup (Senior to individual contributor then back)
            ["Senior Engineer", "Engineer", "Lead Engineer", "Principal Engineer"],
            # Early exit from seniority track
            ["Engineer", "Senior Engineer", "Staff Engineer", "Senior Engineer", "Principal Engineer"],
        ]
        # Weight non-smooth sequences 2:1 over smooth — real careers are messy
        weights = [0.10, 0.10, 0.20, 0.15, 0.25, 0.20]
        sequence = random.choices(TITLE_SEQUENCES, weights=weights, k=1)[0]
        SENIORITY = [1, 2, 3, 3, 4, 5, 6, 7, 8]

        roles = []
        year = start_year
        for i, title in enumerate(sequence):
            duration = random.randint(1, 3)
            # Occasionally add a gap (real career authenticity)
            if random.random() < 0.15:
                year += 1  # 1-year gap between roles
            roles.append(Role(
                title=title,
                company=fake.company(),
                start_year=year,
                end_year=year + duration if i < len(sequence) - 1 else None,
                skills=random.sample(list(TECH_TIMELINE.keys()), k=random.randint(2, 4)),
                claimed_skills_added_date={},
            ))
            year += duration
        return roles

    def _build_timestamps(self, career_start: int, domain: str) -> dict[str, str]:
        """
        Coherent timestamps for real experts.
        Claim starts after tool's inflection year (mainstream adoption) AND after career start.
        Real experts don't claim expert usage from day 1 of a tool's existence.
        """
        timestamps = {}
        for tool, timeline in random.sample(list(TECH_TIMELINE.items()), k=4):
            # Real experts learn tools AFTER inflection (mainstream adoption), not at release
            earliest_valid = max(timeline["inflection_year"], career_start)
            # Claim within 4 years post-inflection, capped at 2025
            claim_year = random.randint(earliest_valid, min(earliest_valid + 4, 2025))
            timestamps[tool] = f"{claim_year}-{random.randint(1,12):02d}"
        return timestamps


    def _build_responses(self, domain: str) -> list[ScreeningResponse]:
        responses = []

        # Core domain: failure story (real expert signal)
        story = random.choice(FAILURE_STORY_BANK)
        latency = random.randint(*story["latency_range"])
        responses.append(ScreeningResponse(
            question_id="q_core_1",
            answer=story["answer"],
            latency_ms=latency,
            topic="core",
            question_difficulty="expert",
        ))

        # Adjacent: openly vague (real expert non-uniformity)
        responses.append(ScreeningResponse(
            question_id="q_adjacent_1",
            answer=random.choice(REAL_ADJACENT_VAGUE),
            latency_ms=random.randint(3000, 8000),
            topic="adjacent",
            question_difficulty="intermediate",
        ))

        return responses


class FraudExpertGenerator:

    def generate(self) -> ExpertProfile:
        profile = ExpertProfile(label="FRAUD")

        career_start = random.randint(2014, 2020)
        profile.employment_history = self._build_smooth_career(career_start)
        profile.skill_timestamps = self._build_inflated_timestamps(career_start)
        profile.screening_responses = self._build_uniform_responses()
        return profile

    def _build_smooth_career(self, start_year: int) -> list[Role]:
        """Perfect monotone upward trajectory — no gaps, no laterals."""
        title_seq = [
            "Junior Software Engineer",
            "Software Engineer",
            "Senior Software Engineer",
            "Staff Software Engineer",
            "Principal Software Engineer",
        ]
        roles = []
        year = start_year
        for i, title in enumerate(title_seq):
            duration = random.randint(1, 2)
            roles.append(Role(
                title=title,
                company=fake.company(),
                start_year=year,
                end_year=year + duration if i < len(title_seq) - 1 else None,
                skills=list(TECH_TIMELINE.keys())[:5],
                claimed_skills_added_date={},
            ))
            year += duration
        return roles

    def _build_inflated_timestamps(self, career_start: int) -> dict[str, str]:
        """40% chance of hard TAV violation, 60% borderline."""
        timestamps = {}
        for tool, timeline in list(TECH_TIMELINE.items())[:5]:
            if random.random() < 0.4:
                # Hard violation: claim before release or at release (impossible expert)
                claim_year = timeline["release_year"] - random.randint(0, 1)
            else:
                # Borderline: exact year of release (very early adopter claim without artifacts)
                claim_year = timeline["release_year"]
            timestamps[tool] = f"{max(claim_year, 2010)}-{random.randint(1,12):02d}"
        return timestamps

    def _build_uniform_responses(self) -> list[ScreeningResponse]:
        """LLM-pattern: uniform fluency regardless of topic, same latency, no failure stories."""
        responses = []
        topics = list(FRAUD_GENERIC_ANSWERS.keys())

        for i, topic in enumerate(topics[:3]):
            responses.append(ScreeningResponse(
                question_id=f"q_{topic}_{i+1}",
                answer=FRAUD_GENERIC_ANSWERS[topic],
                latency_ms=random.randint(2000, 3500),  # Uniform ~AI query time
                topic="core" if topic != "general_adjacent" else "adjacent",
                question_difficulty="expert",
            ))
        return responses


# ─── Profile Generator (used by RL env) ────────────────────────────────────

class ProfileGenerator:

    def __init__(self, profiles: Optional[list[ExpertProfile]] = None,
                 fraud_ratio: float = 0.4):
        self.fraud_ratio = fraud_ratio
        self._profiles = profiles or []
        self._real_gen = RealExpertGenerator()
        self._fraud_gen = FraudExpertGenerator()
        self._pool: list[ExpertProfile] = list(profiles) if profiles else []
        self._pool_idx = 0

    @classmethod
    def from_file(cls, path: str) -> ProfileGenerator:
        with open(path) as f:
            raw = json.load(f)
        profiles = [ExpertProfile.from_dict(d) for d in raw]
        fraud_ratio = sum(1 for p in profiles if p.label == "FRAUD") / len(profiles)
        return cls(profiles=profiles, fraud_ratio=fraud_ratio)

    def sample(self, rng=None) -> tuple[ExpertProfile, str]:
        """Return (profile, true_label). Uses pool if available, else generates live."""
        if self._pool:
            if self._pool_idx >= len(self._pool):
                if rng is not None:
                    perm = rng.permutation(len(self._pool)).tolist()
                    self._pool = [self._pool[i] for i in perm]
                else:
                    random.shuffle(self._pool)
                self._pool_idx = 0
            profile = self._pool[self._pool_idx]
            self._pool_idx += 1
            return profile, profile.label

        # Live generation — use rng to decide fraud/real deterministically
        if rng is not None:
            is_fraud = float(rng.random()) < self.fraud_ratio
        else:
            is_fraud = random.random() < self.fraud_ratio

        if is_fraud:
            profile = self._fraud_gen.generate()
        else:
            profile = self._real_gen.generate()
        return profile, profile.label

    def generate(self, n: int, fraud_ratio: Optional[float] = None) -> list[ExpertProfile]:
        fr = fraud_ratio if fraud_ratio is not None else self.fraud_ratio
        profiles = []
        n_fraud = int(n * fr)
        for _ in range(n_fraud):
            profiles.append(self._fraud_gen.generate())
        for _ in range(n - n_fraud):
            profiles.append(self._real_gen.generate())
        random.shuffle(profiles)
        return profiles


# ─── CLI Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic KIVE expert profiles")
    parser.add_argument("--n", type=int, default=500, help="Total profiles to generate")
    parser.add_argument("--fraud-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="data/synthetic_profiles.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        fake.seed_instance(args.seed)

    gen = ProfileGenerator(fraud_ratio=args.fraud_ratio)
    profiles = gen.generate(args.n, args.fraud_ratio)

    with open(args.output, "w") as f:
        json.dump([p.to_dict() for p in profiles], f, indent=2)

    n_fraud = sum(1 for p in profiles if p.label == "FRAUD")
    print(f"Generated {len(profiles)} profiles: {n_fraud} fraud / {len(profiles) - n_fraud} real")
    print(f"Output: {args.output}")

    if args.verbose:
        sample_real = next(p for p in profiles if p.label == "REAL")
        sample_fraud = next(p for p in profiles if p.label == "FRAUD")
        print("\n--- Sample REAL profile ---")
        print(json.dumps(sample_real.to_dict(), indent=2)[:600])
        print("\n--- Sample FRAUD profile ---")
        print(json.dumps(sample_fraud.to_dict(), indent=2)[:600])
