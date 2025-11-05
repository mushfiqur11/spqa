"""
SPQA (Style Perturbed Question Answering)

An evaluation framework and benchmark for assessing QA performance under 
readability-controlled style perturbations.

This package implements the SPQA framework described in:
"Evaluating Health Question Answering Under Readability-Controlled Style Perturbations"
by Md Mushfiqur Rahman and Kevin Lybarger.

Main Components:
- data_creation: Style transfer and perturbation modules  
- qa_models: Question answering execution and benchmarking
- evaluation: LLM-Judge based evaluation system
- visualization: Results analysis and plotting
- utils: Common utilities and configuration handling
"""

__version__ = "1.0.0"
__author__ = "Md Mushfiqur Rahman and Kevin Lybarger"
__email__ = "mrahma45@gmu.edu, klybarge@gmu.edu"

# Import main classes for easy access
from .data_creation.style_transfer import StyleTransfer
from .qa_models.qa_runner import QARunner
from .evaluation.llm_judge import LLMJudge
from .utils.config import Config

__all__ = [
    "StyleTransfer",
    "QARunner", 
    "LLMJudge",
    "Config",
]
