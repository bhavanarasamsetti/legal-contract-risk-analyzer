"""RAGAS evaluation orchestrator for the legal contract RAG pipeline.

:class:`RagEvaluator` runs the frozen :class:`~app.hybrid_retriever.HybridRetriever`
and :class:`~app.llm.LegalContractLLM` over a manually curated evaluation
dataset, then scores retrieval and generation quality with RAGAS metrics
plus a custom ``document_hit_rate``.

Typical usage::

    from evaluation import RagEvaluator, EvalDataset

    evaluator = RagEvaluator(top_k=5)
    result = evaluator.evaluate("evaluation/datasets/sample.json")
    print(result.to_readme_markdown())
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from app.hybrid_retriever import HybridRetriever
from app.llm import LegalContractLLM

from evaluation.schema import (
    EvalDataset,
    EvalMetrics,
    EvalResult,
    SampleEvalRecord,
)

_DEFAULT_TOP_K = 5

# RAGAS 0.4 column names (see ragas.validation and metric required_columns).
_RAGAS_USER_INPUT = "user_input"
_RAGAS_RESPONSE = "response"
_RAGAS_CONTEXTS = "retrieved_contexts"
_RAGAS_REFERENCE = "reference"


class RagEvaluator:
    """Standalone RAGAS evaluation pipeline.

    Orchestrates hybrid retrieval and LLM generation for each sample in a
    manually curated dataset, then computes:

    - **RAGAS retrieval metrics:** ``context_precision``, ``context_recall``
    - **RAGAS generation metrics:** ``faithfulness``, ``answer_relevancy``,
      ``answer_correctness``
    - **Custom retrieval metric:** ``document_hit_rate`` — fraction of
      samples where ``expected_document`` appears in retrieved sources

    This class does **not**:

    - Modify any frozen application module.
    - Parse PDFs or run ingestion.
    - Call Pinecone or OpenAI directly for the RAG pipeline (delegated to
      frozen collaborators).  RAGAS judge calls are isolated in
      :func:`_compute_ragas_metrics`.

    Example:
        >>> evaluator = RagEvaluator(top_k=5)
        >>> result = evaluator.evaluate("evaluation/datasets/sample.json")
        >>> result.metrics.document_hit_rate
        0.667
    """

    def __init__(
        self,
        top_k: int = _DEFAULT_TOP_K,
        namespace: str = "",
        retriever: HybridRetriever | None = None,
        llm: LegalContractLLM | None = None,
    ) -> None:
        """Initialise the evaluator and its RAG pipeline collaborators.

        Args:
            top_k: Number of chunks retrieved per evaluation question.
            namespace: Pinecone namespace forwarded to
                :class:`~app.hybrid_retriever.HybridRetriever`.
            retriever: Optional pre-built retriever for tests.
            llm: Optional pre-built LLM for tests.  When ``None``, a
                :class:`~app.llm.LegalContractLLM` with ``temperature=0.0``
                is created for deterministic generation.

        Raises:
            ValueError: If ``top_k`` is less than 1.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")

        self._top_k = top_k
        self._retriever = retriever or HybridRetriever(namespace=namespace)
        self._llm = llm or LegalContractLLM(temperature=0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset: EvalDataset | Path | str,
    ) -> EvalResult:
        """Run the full evaluation pipeline on ``dataset``.

        For each sample:

        1. Retrieve context with :class:`~app.hybrid_retriever.HybridRetriever`.
        2. Generate an answer with :class:`~app.llm.LegalContractLLM`.
        3. Record whether the expected document was retrieved.

        Then compute RAGAS metrics over all successful samples and return
        an :class:`~evaluation.schema.EvalResult`.

        Args:
            dataset: An :class:`~evaluation.schema.EvalDataset` instance,
                or a path to a JSON dataset file.

        Returns:
            Aggregated metrics, per-sample records, and README-ready output.

        Raises:
            FileNotFoundError: If a path is given and the file does not exist.
            ValueError: If the dataset fails validation.
            RuntimeError: If RAGAS scoring fails (missing dependency or API
                error).  Individual sample pipeline failures are recorded in
                ``per_sample[].error`` and excluded from RAGAS scoring.
        """
        dataset_path: str | None = None
        if not isinstance(dataset, EvalDataset):
            dataset_path = str(Path(dataset))
            dataset = EvalDataset.load(dataset_path)

        per_sample = self._run_pipeline(dataset)
        document_hit_rate = _document_hit_rate(per_sample)
        ragas_scores = _compute_ragas_metrics(per_sample)
        metrics = _build_metrics(per_sample, ragas_scores, document_hit_rate)
        model = next((row.model for row in per_sample if row.model), "")

        return EvalResult(
            metrics=metrics,
            per_sample=per_sample,
            top_k=self._top_k,
            model=model,
            dataset_path=dataset_path,
            ragas_scores=ragas_scores,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_pipeline(self, dataset: EvalDataset) -> list[SampleEvalRecord]:
        """Execute retrieve → generate for every sample in ``dataset``."""
        records: list[SampleEvalRecord] = []

        for sample in dataset.samples:
            try:
                sources = self._retriever.retrieve(
                    sample.question,
                    top_k=self._top_k,
                )
                contexts = [source["chunk_text"] for source in sources]
                retrieved_documents = sorted(
                    {source["document_name"] for source in sources}
                )
                document_hit = sample.expected_document in retrieved_documents

                if not sources:
                    records.append(
                        SampleEvalRecord(
                            question=sample.question,
                            ground_truth=sample.ground_truth,
                            expected_document=sample.expected_document,
                            retrieved_documents=[],
                            document_hit=False,
                            error="No contexts retrieved.",
                        )
                    )
                    continue

                llm_response = self._llm.complete(sample.question, context=sources)
                records.append(
                    SampleEvalRecord(
                        question=sample.question,
                        ground_truth=sample.ground_truth,
                        expected_document=sample.expected_document,
                        answer=llm_response["answer"],
                        contexts=contexts,
                        retrieved_documents=retrieved_documents,
                        document_hit=document_hit,
                        model=llm_response["model"],
                    )
                )
            except Exception as exc:
                print(f"\nERROR on question:\n{sample.question}")
                print(exc)

                records.append(
                    SampleEvalRecord(
                        question=sample.question,
                        ground_truth=sample.ground_truth,
                        expected_document=sample.expected_document,
                        error=str(exc),
                    )
                )

        return records


def _document_hit_rate(records: list[SampleEvalRecord]) -> float:
    """Fraction of samples where the expected document was retrieved."""
    if not records:
        return 0.0
    hits = sum(1 for row in records if row.document_hit)
    return hits / len(records)


def _successful_records(records: list[SampleEvalRecord]) -> list[SampleEvalRecord]:
    """Return records where the pipeline completed without error."""
    return [row for row in records if row.error is None and row.answer]


def _to_ragas_rows(records: list[SampleEvalRecord]) -> list[dict[str, Any]]:
    """Convert successful pipeline records to RAGAS 0.4 dataset rows."""
    return [
        {
            _RAGAS_USER_INPUT: row.question,
            _RAGAS_RESPONSE: row.answer,
            _RAGAS_CONTEXTS: row.contexts,
            _RAGAS_REFERENCE: row.ground_truth,
        }
        for row in records
    ]


def _compute_ragas_metrics(records: list[SampleEvalRecord]) -> dict[str, float]:
    """Score successful samples with RAGAS and return mean metric values.

    RAGAS uses a separate judge LLM and embeddings model configured from
    :func:`~app.config.get_config`.  This is isolated here so the rest of
    the evaluator never calls OpenAI directly.

    Raises:
        RuntimeError: If RAGAS is not installed, no successful samples exist,
            or the RAGAS evaluation call fails.
    """
    successful = _successful_records(records)
    if not successful:
        return {}

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        print("\nREAL IMPORT ERROR:")
        print(repr(exc))
        raise

    from app.config import get_config

    config = get_config()
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=config.openai_chat_model,
            api_key=config.openai_api_key,
            temperature=0.0,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=config.openai_embedding_model,
            api_key=config.openai_api_key,
        )
    )

    ragas_dataset = Dataset.from_list(_to_ragas_rows(successful))
    metrics = [
        faithfulness,
        answer_relevancy,
        answer_correctness,
        context_precision,
        context_recall,
    ]

    try:
        result = evaluate(
            ragas_dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            raise_exceptions=True,
        )
    except Exception as exc:
        raise RuntimeError(f"RAGAS evaluation failed: {exc}") from exc

    return _extract_ragas_means(result)


def _extract_ragas_means(result: Any) -> dict[str, float]:
    """Extract mean metric scores from a RAGAS EvaluationResult."""
    scores: dict[str, float] = {}
    repr_dict = getattr(result, "_repr_dict", {})
    for name, value in repr_dict.items():
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        scores[name] = numeric
    return scores


def _build_metrics(
    records: list[SampleEvalRecord],
    ragas_scores: dict[str, float],
    document_hit_rate: float,
) -> EvalMetrics:
    """Assemble :class:`~evaluation.schema.EvalMetrics` from raw scores."""
    successful = _successful_records(records)
    return EvalMetrics(
        faithfulness=ragas_scores.get("faithfulness"),
        answer_relevancy=ragas_scores.get("answer_relevancy"),
        answer_correctness=ragas_scores.get("answer_correctness"),
        context_precision=ragas_scores.get("context_precision"),
        context_recall=ragas_scores.get("context_recall"),
        document_hit_rate=document_hit_rate,
        num_samples=len(records),
        num_successful=len(successful),
    )
