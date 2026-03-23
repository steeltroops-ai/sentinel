# services/svp/detector.py
# Specificity Variance Profile — weight 0.24
# Key insight: measure VARIANCE of specificity across topics, not mean.
# Real experts are hyper-specific in domain, openly vague outside it.
# AI responses are uniformly fluent everywhere — model doesn't know what it doesn't know.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kive.shared.schemas import FlagDetail, ProbeSuggestion, SignalRequest

WEIGHT = 0.24

# Technical vocabulary per domain — used for term density scoring
TECH_VOCAB: dict[str, set[str]] = {
    "ml": {
        "gradient", "backpropagation", "overfitting", "regularization", "dropout",
        "batch normalization", "learning rate", "epoch", "loss function", "tensor",
        "embedding", "attention", "transformer", "fine-tuning", "inference",
        "precision", "recall", "f1", "roc", "auc", "cross-entropy", "softmax",
        "conv", "pooling", "lstm", "encoder", "decoder", "latent", "tokenizer",
    },
    "devops": {
        "kubernetes", "container", "pod", "deployment", "ingress", "helm", "kubectl",
        "namespace", "configmap", "secret", "persistent volume", "statefulset",
        "docker", "image", "registry", "cicd", "pipeline", "terraform", "ansible",
        "prometheus", "grafana", "alerting", "replica", "autoscaling", "hpa",
    },
    "backend": {
        "async", "await", "coroutine", "deadlock", "race condition", "mutex",
        "connection pool", "transaction", "index", "query plan", "n+1", "cache",
        "ttl", "eviction", "circuit breaker", "retry", "idempotent", "rest",
        "grpc", "serialization", "middleware", "rate limit", "auth", "jwt",
    },
    "data": {
        "partition", "shuffle", "join", "groupby", "aggregation", "window function",
        "materialized view", "schema evolution", "parquet", "avro", "delta lake",
        "lineage", "dbt", "dag", "airflow", "spark", "broadcast", "skew",
        "cardinality", "null handling", "upsert", "scd", "slowly changing",
    },
}

# Patterns indicating numerical concreteness (specific counts, versions, dates)
_NUMERIC_PATTERN = re.compile(
    r"\b\d+\.\d+|\bv\d+\b|\b\d{4}\b|\b\d+%|\b\d+ms|\b\d+[kmg]b\b|\b\d+ (seconds|minutes|hours|days)\b",
    re.IGNORECASE,
)

# Opinionated language markers -- real experts have preferences and take sides
_OPINION_PATTERNS = [
    re.compile(r"\b(I prefer|I always use|I never use|I stopped using|my go-to|in my experience)\b", re.IGNORECASE),
    re.compile(r"\b(better than|worse than|overrated|underrated|honestly|frankly|unpopular opinion)\b", re.IGNORECASE),
    re.compile(r"\b(I disagree|I don't think|I wouldn't recommend|avoid using|we switched from)\b", re.IGNORECASE),
    re.compile(r"\b(the problem with|the downside of|the tradeoff is|not great for|overkill for)\b", re.IGNORECASE),
]

# Hedging phrases -- GPT over-hedges on factual questions
_HEDGING_PATTERNS = re.compile(
    r"\b(it depends|generally speaking|in most cases|it's worth noting|one could argue|"
    r"there are various|it is important to|when it comes to|one of the key|broadly speaking|"
    r"typically|arguably|essentially|fundamentally|it varies)\b",
    re.IGNORECASE,
)


@dataclass
class SVPResult:
    score: float           # [0,1] — high = uniform fluency = fraud signal
    confidence: float
    variance: float        # Raw specificity variance across topics
    per_topic_scores: dict[str, float] = field(default_factory=dict)
    flags: list[FlagDetail] = field(default_factory=list)
    probe_suggestion: Optional[ProbeSuggestion] = None


class SVPDetector:
    """
    Specificity Variance Profile detector.
    Computes per-topic specificity (NER density + technical term density + numerical
    concreteness), then measures VARIANCE. Low variance = fraud signal.
    """

    def __init__(self):
        self._nlp = None
        self._initialized = False

    async def initialize(self):
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        except Exception:
            self._nlp = None
        self._initialized = True

    async def close(self):
        self._nlp = None

    async def analyze(self, request: SignalRequest) -> SVPResult:
        if not self._initialized:
            await self.initialize()

        responses = request.screening_responses
        if not responses:
            return SVPResult(score=0.5, confidence=0.2, variance=0.0)

        topic_scores: dict[str, float] = {}

        for resp in responses:
            topic = resp.topic or "general"
            answer = resp.answer or ""
            if not answer.strip():
                continue
            score = self._specificity_score(answer, topic)
            topic_scores[topic] = score

        if len(topic_scores) < 2:
            # Can't compute variance with 1 topic — low confidence
            scores = list(topic_scores.values())
            mean_score = float(np.mean(scores)) if scores else 0.5
            return SVPResult(
                score=0.3,
                confidence=0.2,
                variance=0.0,
                per_topic_scores=topic_scores,
            )

        scores_arr = np.array(list(topic_scores.values()))
        variance = float(np.var(scores_arr))
        mean = float(np.mean(scores_arr))

        # Low variance AND high mean specificity = uniformly fluent = fraud
        # Low variance AND low mean = uniformly vague = also suspicious but different
        # High variance = real expert pattern (expert in domain, vague outside)
        if variance < 0.02:
            # Very uniform — fraud signal
            fraud_score = min(0.70 + (mean * 0.30), 1.0)
            confidence = 0.78
        elif variance < 0.05:
            fraud_score = 0.50
            confidence = 0.55
        else:
            # Good variance — real expert pattern
            fraud_score = max(0.10, 0.30 - variance * 2)
            confidence = 0.65

        fraud_score = float(np.clip(fraud_score, 0.0, 1.0))

        flags = []
        probe = None

        if variance < 0.03 and len(topic_scores) >= 2:
            flags.append(FlagDetail(
                type="svp_uniform_fluency",
                description=(
                    f"Specificity variance {variance:.4f} across {len(topic_scores)} topics. "
                    f"Real experts show non-uniform specificity — expert in domain, vague outside. "
                    f"Uniform fluency is a strong AI-assisted response pattern."
                ),
                severity="high" if variance < 0.01 else "medium",
                evidence={
                    "variance": round(variance, 4),
                    "mean_specificity": round(mean, 3),
                    "n_topics": len(topic_scores),
                    "per_topic": {k: round(v, 3) for k, v in topic_scores.items()},
                },
            ))
            probe = ProbeSuggestion(
                question=(
                    "I'm going to shift topics — can you tell me something you "
                    "genuinely don't know well in your field, and why?"
                ),
                target_dimension="SVP",
                expected_fraud_response_pattern=(
                    "Gives a fluent, humble-sounding answer that still sounds competent. "
                    "Real experts struggle or become noticeably vaguer when asked this."
                ),
            )

        return SVPResult(
            score=round(fraud_score, 3),
            confidence=round(confidence, 3),
            variance=round(variance, 4),
            per_topic_scores={k: round(v, 3) for k, v in topic_scores.items()},
            flags=flags,
            probe_suggestion=probe,
        )

    def _specificity_score(self, text: str, topic: str) -> float:
        """Compute specificity [0,1] for a single response."""
        words = text.split()
        n_words = max(len(words), 1)

        # 1. NER entity density (SpaCy)
        ner_density = 0.0
        if self._nlp:
            try:
                doc = self._nlp(text[:1000])  # cap for performance
                relevant_labels = {"ORG", "PRODUCT", "GPE", "DATE", "CARDINAL", "QUANTITY"}
                ner_hits = sum(1 for e in doc.ents if e.label_ in relevant_labels)
                ner_density = min(ner_hits / (n_words / 20), 1.0)
            except Exception:
                pass

        # 2. Technical term density (domain-specific vocabulary)
        domain = self._infer_domain(topic)
        vocab = TECH_VOCAB.get(domain, set())
        text_lower = text.lower()
        tech_hits = sum(1 for term in vocab if term in text_lower)
        tech_density = min(tech_hits / max(len(vocab) * 0.05, 1), 1.0)

        # 3. Numerical concreteness (specific numbers, versions, durations)
        numeric_hits = len(_NUMERIC_PATTERN.findall(text))
        numeric_density = min(numeric_hits / (n_words / 30), 1.0)

        # 4. Proper noun density (NEW) -- real experts name specific things
        proper_noun_density = 0.0
        if self._nlp:
            try:
                doc = self._nlp(text[:1000])
                # Count PROPN tokens excluding sentence-initial position
                proper_nouns = sum(
                    1 for token in doc
                    if token.pos_ == "PROPN" and token.i > 0
                    and doc[token.i - 1].text != "."
                )
                proper_noun_density = min(proper_nouns / (n_words / 15), 1.0)
            except Exception:
                pass

        # 5. Opinionated language score (NEW) -- real experts take positions
        opinion_hits = sum(1 for p in _OPINION_PATTERNS if p.search(text))
        opinion_score = min(opinion_hits / 2.0, 1.0)  # 2+ opinion markers = max

        # 6. Hedging density (NEW) -- GPT over-hedges; inverse signal
        hedging_hits = len(_HEDGING_PATTERNS.findall(text))
        hedging_penalty = min(hedging_hits / (n_words / 50), 1.0)  # high hedging = low specificity

        # Weighted composite: original 3 signals + 3 new signals
        score = (
            0.20 * ner_density +
            0.25 * tech_density +
            0.15 * numeric_density +
            0.15 * proper_noun_density +
            0.15 * opinion_score +
            0.10 * (1.0 - hedging_penalty)  # invert: less hedging = more specificity
        )
        return float(np.clip(score, 0.0, 1.0))

    def _infer_domain(self, topic: str) -> str:
        topic_lower = topic.lower()
        if any(k in topic_lower for k in ["ml", "model", "neural", "prediction", "torch", "tensorflow"]):
            return "ml"
        if any(k in topic_lower for k in ["k8s", "kube", "docker", "deploy", "infra", "devops", "ci"]):
            return "devops"
        if any(k in topic_lower for k in ["data", "spark", "pipeline", "etl", "warehouse"]):
            return "data"
        return "backend"
