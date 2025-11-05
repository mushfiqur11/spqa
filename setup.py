#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="spqa",
    version="1.0.0",
    author="Md Mushfiqur Rahman and Kevin Lybarger",
    author_email="mrahma45@gmu.edu, klybarge@gmu.edu",
    description="Style Perturbed Question Answering (SPQA): Evaluating Health Question Answering Under Readability-Controlled Style Perturbations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mushfiqur11/spqa",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "spqa-style-transfer=scripts.run_style_transfer:main",
            "spqa-qa-benchmark=scripts.run_qa_benchmark:main",
            "spqa-llm-judge=scripts.run_llm_judge:main",
            "spqa-generate-results=scripts.generate_results:main",
        ],
    },
)
