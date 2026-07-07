"""RAGAS evaluation package for the legal contract RAG pipeline.

Public exports::

    from evaluation import RagEvaluator, EvalDataset, EvalResult
"""

from evaluation.rag_evaluator import RagEvaluator
from evaluation.schema import (
    EvalDataset,
    EvalMetrics,
    EvalResult,
    EvalSample,
    SampleEvalRecord,
)

__all__ = [
    "EvalDataset",
    "EvalMetrics",
    "EvalResult",
    "EvalSample",
    "RagEvaluator",
    "SampleEvalRecord",
]
