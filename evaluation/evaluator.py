"""
Evaluation Module
=================
Defines and computes evaluation metrics for the Weather & News AI Assistant.

Metrics implemented
-------------------
1. Keyword Presence Score  (automated, 0–1)
   Checks that expected keywords appear and unexpected keywords are absent.

2. Length Adequacy Score  (automated, 0–1)
   Verifies the response is at least `min_length` characters long.

3. Criteria Coverage Score  (automated keyword heuristic, 0–1)
   For each named criterion, uses a keyword-level heuristic to guess
   whether it is satisfied.

4. LLM Judge Score  (AI-based, 1–5 per sub-dimension)
   Uses Claude to rate the response on: Relevance, Accuracy,
   Completeness, and Clarity.  Requires an Anthropic API key.

5. Composite Score  (0–100)
   Weighted combination of all the above.

Usage
-----
    from evaluation.evaluator import Evaluator, run_evaluation_async

    evaluator = Evaluator(api_key="sk-ant-...", model="claude-haiku-4-5-20251001")

    # Evaluate a single response
    result = evaluator.evaluate_response(test_case, agent_response)

    # Run the full eval suite (async)
    results = await run_evaluation_async(orchestrator, evaluator, test_cases)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

# ── Load the evaluation dataset ───────────────────────────────────────────────
_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


def load_eval_dataset() -> list[dict]:
    with open(_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class CriterionResult:
    name: str
    description: str
    passed: bool
    score: float  # 0.0 or 1.0 for binary, 0–1 for partial


@dataclass
class EvalResult:
    test_id: str
    category: str
    query: str
    response: str
    latency_seconds: float

    # Automated metrics
    keyword_presence_score: float = 0.0   # 0–1
    length_adequacy_score: float = 0.0    # 0–1
    criteria_coverage_score: float = 0.0  # 0–1
    criteria_results: list[CriterionResult] = field(default_factory=list)

    # LLM judge metrics (1–5 each, -1 if not run)
    llm_relevance: float = -1.0
    llm_accuracy: float = -1.0
    llm_completeness: float = -1.0
    llm_clarity: float = -1.0
    llm_judge_raw: str = ""

    # Composite (0–100)
    composite_score: float = 0.0

    # Error info
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "category": self.category,
            "query": self.query,
            "response_preview": self.response[:200] + "…" if len(self.response) > 200 else self.response,
            "latency_seconds": round(self.latency_seconds, 2),
            "keyword_presence_score": round(self.keyword_presence_score, 3),
            "length_adequacy_score": round(self.length_adequacy_score, 3),
            "criteria_coverage_score": round(self.criteria_coverage_score, 3),
            "llm_relevance": self.llm_relevance,
            "llm_accuracy": self.llm_accuracy,
            "llm_completeness": self.llm_completeness,
            "llm_clarity": self.llm_clarity,
            "composite_score": round(self.composite_score, 1),
            "error": self.error,
        }


# ── Scoring weights ────────────────────────────────────────────────────────────

WEIGHTS = {
    "keyword_presence": 0.20,
    "length_adequacy": 0.10,
    "criteria_coverage": 0.30,
    "llm_judge": 0.40,   # averaged over 4 sub-dimensions; set to 0 if not run
}


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Computes multiple evaluation metrics for generated responses.

    Args:
        api_key: Anthropic API key (required for LLM judge).
        model: Claude model to use as judge.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
    ):
        self.api_key = api_key
        self.model = model
        self._client: anthropic.Anthropic | None = (
            anthropic.Anthropic(api_key=api_key) if api_key else None
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate_response(
        self,
        test_case: dict,
        response: str,
        latency: float = 0.0,
        run_llm_judge: bool = True,
    ) -> EvalResult:
        """
        Evaluate a single response against a test case.

        Returns an EvalResult with all metric scores populated.
        """
        result = EvalResult(
            test_id=test_case["id"],
            category=test_case["category"],
            query=test_case["query"],
            response=response,
            latency_seconds=latency,
        )

        # ── 1. Keyword presence score ──────────────────────────────────────────
        result.keyword_presence_score = self._keyword_presence(
            response,
            test_case.get("expected_keywords", []),
            test_case.get("unexpected_keywords", []),
        )

        # ── 2. Length adequacy score ───────────────────────────────────────────
        min_len = test_case.get("min_length", 50)
        result.length_adequacy_score = min(1.0, len(response) / max(min_len, 1))

        # ── 3. Criteria coverage score ─────────────────────────────────────────
        criteria = test_case.get("criteria", {})
        crit_results = []
        for name, description in criteria.items():
            passed = self._heuristic_criterion(name, description, response)
            crit_results.append(
                CriterionResult(
                    name=name,
                    description=description,
                    passed=passed,
                    score=1.0 if passed else 0.0,
                )
            )
        result.criteria_results = crit_results
        result.criteria_coverage_score = (
            sum(c.score for c in crit_results) / len(crit_results)
            if crit_results else 0.0
        )

        # ── 4. LLM judge score ─────────────────────────────────────────────────
        if run_llm_judge and self._client is not None:
            try:
                self._run_llm_judge(result, test_case)
            except Exception as exc:
                result.llm_judge_raw = f"LLM judge failed: {exc}"

        # ── 5. Composite score ─────────────────────────────────────────────────
        result.composite_score = self._composite(result)

        return result

    # ── Automated metrics ──────────────────────────────────────────────────────

    @staticmethod
    def _keyword_presence(
        response: str,
        expected: list[str],
        unexpected: list[str],
    ) -> float:
        """Score based on expected keyword hits minus unexpected keyword hits."""
        lower = response.lower()

        if not expected and not unexpected:
            return 1.0

        hits = sum(1 for kw in expected if kw.lower() in lower)
        misses = sum(1 for kw in unexpected if kw.lower() in lower)

        presence_score = hits / max(len(expected), 1)
        penalty = misses / max(len(unexpected), 1)

        return max(0.0, presence_score - 0.5 * penalty)

    @staticmethod
    def _heuristic_criterion(name: str, description: str, response: str) -> bool:
        """
        Simple heuristic: look for topic-related keywords in the response.
        This is imperfect but fast and requires no API calls.
        """
        lower_resp = response.lower()
        lower_desc = description.lower()
        lower_name = name.lower()

        # Map criterion names to indicator keywords
        keyword_map: dict[str, list[str]] = {
            "mentions_city": ["london", "tokyo", "new york", "paris", "berlin",
                              "sydney", "nyc", "uk", "japan", "france", "germany",
                              "australia", "usa", "city"],
            "has_temperature": ["°c", "celsius", "degree", "temperature", "temp",
                                "°f", "fahrenheit", "hot", "cold", "warm", "cool"],
            "has_conditions": ["clear", "cloudy", "rain", "snow", "fog", "sun",
                               "wind", "storm", "overcast", "shower", "drizzle",
                               "partly", "mainly", "sky"],
            "has_forecast": ["forecast", "tomorrow", "day", "monday", "tuesday",
                             "wednesday", "thursday", "friday", "saturday", "sunday",
                             "week", "days", "daily"],
            "mentions_rain": ["rain", "precipitation", "shower", "drizzle",
                              "probability", "mm", "wet"],
            "no_error": [],  # special case — checked separately
            "no_crash": [],  # special case
            "has_multiple_stories": [],  # checked by count
            "has_titles": ["story", "article", "—", ":", "•", "-"],
            "has_links": ["http", "www.", "hn.algolia", "hacker", "news.ycombinator"],
            "topic_relevance": [],
            "has_stories": ["story", "article", "news", "post"],
            "has_weather": ["temperature", "°c", "weather", "cloudy", "sunny"],
            "has_news": ["news", "story", "article", "hn", "hacker"],
            "has_forecast": ["forecast", "days", "daily", "week"],
            "has_climate_news": ["climate", "environment", "carbon", "emission",
                                 "renewable", "weather change", "global warming"],
            "handles_gracefully": ["not found", "location", "city", "invalid",
                                   "could not", "cannot find", "sorry", "unable"],
            "has_multiple_stories": [],
        }

        # Special cases
        if name in ("no_error", "no_crash"):
            bad_words = ["traceback", "exception", "error:", "fatal", "crash"]
            return not any(bw in lower_resp for bw in bad_words)

        if name == "has_multiple_stories":
            # Count bullet points, numbered items, or "•" characters
            bullets = len(re.findall(r"(?:^|\n)\s*[-•*\d\.]+\s+\S", response))
            return bullets >= 2

        if name == "topic_relevance":
            # Grab keywords from the description itself
            words = re.findall(r"\b[a-zA-Z]{4,}\b", lower_desc)
            return any(w in lower_resp for w in words)

        indicators = keyword_map.get(name, [])
        if not indicators:
            # Fallback: pull keywords from the description
            words = re.findall(r"\b[a-zA-Z]{4,}\b", lower_desc)
            return any(w in lower_resp for w in words)

        return any(ind in lower_resp for ind in indicators)

    # ── LLM judge ──────────────────────────────────────────────────────────────

    JUDGE_PROMPT = """You are an objective AI evaluator assessing the quality of
an AI assistant's response to a user query about weather or news.

Query: {query}

Response:
{response}

Rate the response on each of the following dimensions (integer 1–5):
- Relevance (1=completely off-topic, 5=directly addresses the query)
- Accuracy (1=clearly wrong/hallucinated, 5=factually plausible and well-grounded)
- Completeness (1=major gaps, 5=comprehensive and thorough)
- Clarity (1=confusing, 5=clear, well-structured, easy to read)

Respond ONLY with valid JSON in this exact format (no extra text):
{{"relevance": <int>, "accuracy": <int>, "completeness": <int>, "clarity": <int>, "reasoning": "<one sentence>"}}
"""

    def _run_llm_judge(self, result: EvalResult, test_case: dict) -> None:
        """Call Claude to score the response on 4 dimensions."""
        assert self._client is not None

        prompt = self.JUDGE_PROMPT.format(
            query=result.query,
            response=result.response[:1500],  # truncate to keep tokens manageable
        )

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = resp.content[0].text.strip()
        result.llm_judge_raw = raw

        # Parse JSON
        # Strip markdown code fences if present
        raw_clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        scores = json.loads(raw_clean)

        result.llm_relevance = float(scores.get("relevance", 3))
        result.llm_accuracy = float(scores.get("accuracy", 3))
        result.llm_completeness = float(scores.get("completeness", 3))
        result.llm_clarity = float(scores.get("clarity", 3))

    # ── Composite score ────────────────────────────────────────────────────────

    @staticmethod
    def _composite(result: EvalResult) -> float:
        """
        Weighted composite score on a 0–100 scale.

        Weights (see WEIGHTS at module level):
          keyword_presence  20 %
          length_adequacy   10 %
          criteria_coverage 30 %
          llm_judge         40 %  (average of 4 dimensions normalised to 0–1)
        """
        score = 0.0
        score += WEIGHTS["keyword_presence"] * result.keyword_presence_score
        score += WEIGHTS["length_adequacy"] * result.length_adequacy_score
        score += WEIGHTS["criteria_coverage"] * result.criteria_coverage_score

        # LLM judge: average of 4 dims, normalised from [1,5] → [0,1]
        llm_scores = [
            result.llm_relevance,
            result.llm_accuracy,
            result.llm_completeness,
            result.llm_clarity,
        ]
        valid_llm = [s for s in llm_scores if s >= 1]
        if valid_llm:
            llm_avg_01 = (sum(valid_llm) / len(valid_llm) - 1) / 4  # [1,5]→[0,1]
            score += WEIGHTS["llm_judge"] * llm_avg_01
        else:
            # Redistribute LLM weight to criteria_coverage if judge wasn't run
            extra = WEIGHTS["llm_judge"] * result.criteria_coverage_score
            score += extra

        return round(score * 100, 1)


# ── Async runner ───────────────────────────────────────────────────────────────

async def run_evaluation_async(
    orchestrator,
    evaluator: Evaluator,
    test_cases: list[dict] | None = None,
    run_llm_judge: bool = True,
    progress_callback=None,
) -> list[EvalResult]:
    """
    Run the full evaluation suite asynchronously.

    Args:
        orchestrator: WeatherNewsOrchestrator instance.
        evaluator: Evaluator instance.
        test_cases: List of test-case dicts (defaults to the bundled dataset).
        run_llm_judge: Whether to include the LLM judge (slower, costs tokens).
        progress_callback: Optional callable(current, total, test_id) for UI updates.

    Returns:
        List of EvalResult objects.
    """
    if test_cases is None:
        test_cases = load_eval_dataset()

    results: list[EvalResult] = []

    for i, tc in enumerate(test_cases):
        if progress_callback:
            progress_callback(i, len(test_cases), tc["id"])

        t0 = time.perf_counter()
        try:
            agent_result = await orchestrator.process_query(tc["query"])
            response_text = agent_result.get("response", "")
            error = agent_result.get("error")
        except Exception as exc:
            response_text = f"[Orchestrator error] {exc}"
            error = str(exc)
        latency = time.perf_counter() - t0

        eval_result = evaluator.evaluate_response(
            tc,
            response_text,
            latency=latency,
            run_llm_judge=run_llm_judge,
        )
        if error and not eval_result.error:
            eval_result.error = error

        results.append(eval_result)

    if progress_callback:
        progress_callback(len(test_cases), len(test_cases), "done")

    return results


# ── Summary helpers ────────────────────────────────────────────────────────────

def summarise_results(results: list[EvalResult]) -> dict[str, Any]:
    """Return aggregate statistics across all eval results."""
    if not results:
        return {}

    n = len(results)
    by_category: dict[str, list[EvalResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    def avg(vals):
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    return {
        "total_cases": n,
        "avg_composite_score": avg([r.composite_score for r in results]),
        "avg_keyword_presence": avg([r.keyword_presence_score for r in results]),
        "avg_length_adequacy": avg([r.length_adequacy_score for r in results]),
        "avg_criteria_coverage": avg([r.criteria_coverage_score for r in results]),
        "avg_llm_relevance": avg([r.llm_relevance for r in results if r.llm_relevance >= 1]),
        "avg_llm_accuracy": avg([r.llm_accuracy for r in results if r.llm_accuracy >= 1]),
        "avg_llm_completeness": avg([r.llm_completeness for r in results if r.llm_completeness >= 1]),
        "avg_llm_clarity": avg([r.llm_clarity for r in results if r.llm_clarity >= 1]),
        "avg_latency_seconds": avg([r.latency_seconds for r in results]),
        "error_count": sum(1 for r in results if r.error),
        "by_category": {
            cat: {
                "count": len(rs),
                "avg_composite": avg([r.composite_score for r in rs]),
            }
            for cat, rs in by_category.items()
        },
    }
