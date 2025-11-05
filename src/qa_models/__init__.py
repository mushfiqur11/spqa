"""
Question Answering modules for SPQA framework.

This module provides functionality for running QA models on style-transferred questions
and benchmarking their performance across different linguistic variants.

Main Components:
- qa_runner: Main QA execution engine
- benchmark: Benchmarking utilities and metrics
"""

from .qa_runner import QARunner
from .benchmark import QABenchmark

__all__ = [
    "QARunner",
    "QABenchmark",
]
