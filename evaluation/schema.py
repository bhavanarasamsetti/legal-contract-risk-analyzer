"""Pydantic schemas for the RAGAS evaluation pipeline."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EvalSample(BaseModel):
    """One manually authored evaluation example.

    Attributes:
        question: Natural-language question posed to the RAG pipeline.
        ground_truth: Reference answer used by RAGAS generation metrics.
        expected_document: PDF filename that should appear in retrieved
            sources (e.g. ``"nda.pdf"``).  Used for the custom
            ``document_hit_rate`` retrieval metric.
    """

    question: str = Field(min_length=1)
    ground_truth: str = Field(min_length=1)
    expected_document: str = Field(min_length=1)

    @field_validator("question", "ground_truth", "expected_document")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty or blank.")
        return stripped


class EvalDataset(BaseModel):
    """A versioned collection of evaluation samples loaded from JSON.

    Attributes:
        version: Schema version number for forward compatibility.
        description: Optional human-readable description of the dataset.
        samples: Non-empty list of :class:`EvalSample` records.
    """

    version: int = 1
    description: str = ""
    samples: list[EvalSample] = Field(min_length=1)

    @classmethod
    def load(cls, path: Path | str) -> "EvalDataset":
        """Load and validate a dataset from a JSON file.

        Args:
            path: Path to a JSON file with ``version``, optional
                ``description``, and ``samples`` keys.

        Returns:
            A validated :class:`EvalDataset` instance.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file is malformed or fails validation.
        """
        import json

        dataset_path = Path(path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

        with dataset_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)

        if not isinstance(payload, dict):
            raise ValueError("Evaluation dataset must be a JSON object.")

        return cls.model_validate(payload)


class SampleEvalRecord(BaseModel):
    """Pipeline output for a single evaluation sample.

    Attributes:
        question: The evaluation question.
        ground_truth: Reference answer from the dataset.
        expected_document: Expected PDF filename.
        answer: LLM-generated answer (empty if the pipeline failed).
        contexts: Retrieved chunk texts passed to the LLM.
        retrieved_documents: Unique document names from retrieval.
        document_hit: Whether ``expected_document`` appears in retrieval.
        model: OpenAI model ID used for generation, or empty on failure.
        error: Error message when pipeline execution failed for this row.
    """

    question: str
    ground_truth: str
    expected_document: str
    answer: str = ""
    contexts: list[str] = Field(default_factory=list)
    retrieved_documents: list[str] = Field(default_factory=list)
    document_hit: bool = False
    model: str = ""
    error: str | None = None


class EvalMetrics(BaseModel):
    """Aggregated evaluation metrics for README and reporting.

    RAGAS metrics are floats in ``[0.0, 1.0]`` (higher is better).
    ``document_hit_rate`` is the fraction of samples where the expected
    document appeared in retrieved sources.

    Attributes:
        faithfulness: Are answers grounded in retrieved context?
        answer_relevancy: Are answers relevant to the question?
        answer_correctness: Do answers match the reference answer?
        context_precision: Are relevant chunks ranked higher?
        context_recall: Do retrieved contexts cover the reference answer?
        document_hit_rate: Custom retrieval metric (0.0–1.0).
        num_samples: Total samples evaluated.
        num_successful: Samples where the pipeline completed without error.
    """

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    answer_correctness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    document_hit_rate: float = Field(ge=0.0, le=1.0)
    num_samples: int = Field(ge=0)
    num_successful: int = Field(ge=0)


class EvalResult(BaseModel):
    """Complete output of a RAGAS evaluation run.

    Attributes:
        metrics: Aggregated scores suitable for README reporting.
        per_sample: Per-question pipeline outputs for inspection and tracing.
        top_k: Retrieval depth used during the run.
        model: Generation model ID (from the first successful sample).
        evaluated_at: UTC timestamp of the evaluation run (ISO 8601).
        dataset_path: Path to the dataset file, if one was loaded from disk.
        ragas_scores: Raw per-metric mean scores returned by RAGAS.
    """

    metrics: EvalMetrics
    per_sample: list[SampleEvalRecord]
    top_k: int
    model: str = ""
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    dataset_path: str | None = None
    ragas_scores: dict[str, float] = Field(default_factory=dict)

    def to_readme_markdown(self) -> str:
        """Format aggregated metrics as a Markdown table for README inclusion."""
        def _fmt(value: float | None) -> str:
            return f"{value:.3f}" if value is not None else "N/A"

        lines = [
            "| Metric | Score |",
            "|---|---:|",
            f"| Faithfulness | {_fmt(self.metrics.faithfulness)} |",
            f"| Answer relevancy | {_fmt(self.metrics.answer_relevancy)} |",
            f"| Answer correctness | {_fmt(self.metrics.answer_correctness)} |",
            f"| Context precision | {_fmt(self.metrics.context_precision)} |",
            f"| Context recall | {_fmt(self.metrics.context_recall)} |",
            f"| Document hit rate | {self.metrics.document_hit_rate:.3f} |",
            "",
            f"Samples: {self.metrics.num_successful}/{self.metrics.num_samples} successful  ",
            f"top_k={self.top_k}  model={self.model or 'N/A'}  ",
            f"evaluated_at={self.evaluated_at}",
        ]
        return "\n".join(lines)

    def to_langsmith_dict(self) -> dict[str, Any]:
        """Serialize to a LangSmith-compatible evaluation summary dict."""
        return {
            "metrics": self.metrics.model_dump(),
            "per_sample": [row.model_dump() for row in self.per_sample],
            "top_k": self.top_k,
            "model": self.model,
            "evaluated_at": self.evaluated_at,
            "dataset_path": self.dataset_path,
        }
