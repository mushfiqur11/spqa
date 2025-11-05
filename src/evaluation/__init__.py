"""
Evaluation modules for SPQA framework.

This module provides the LLM-Judge evaluation system for assessing QA performance
across the four key criteria defined in the SPQA paper.

Main Components:
- llm_judge: Main LLM-Judge implementation
- rubrics: Evaluation rubrics and criteria definitions
- metrics: Evaluation metrics and scoring
- validation: Judge validation utilities
"""

from .llm_judge import LLMJudge
from .rubrics import EvaluationRubrics
from .metrics import EvaluationMetrics
from .validation import JudgeValidator

__all__ = [
    "LLMJudge",
    "EvaluationRubrics",
    "EvaluationMetrics",
    "JudgeValidator",
]
