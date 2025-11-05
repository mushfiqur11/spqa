"""
Benchmarking utilities for SPQA framework.

This module provides utilities for benchmarking QA model performance across
different style variants, including statistical analysis and comparison tools.
"""

import statistics
from typing import Dict, List, Any, Optional, Tuple
from datasets import DatasetDict
import pandas as pd

from ..utils.config import Config
from ..utils.data_utils import DataUtils


class QABenchmark:
    """
    Benchmarking utilities for QA performance analysis.
    
    This class provides tools for comparing QA model performance across
    different style variants and generating benchmark reports.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize QABenchmark.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or Config()
    
    def compare_models(
        self,
        model_results: Dict[str, DatasetDict],
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Compare performance across multiple QA models.
        
        Args:
            model_results: Dictionary mapping model names to their results
            output_dir: Output directory for comparison results
            
        Returns:
            Comprehensive comparison analysis
        """
        print("Comparing QA model performance...")
        
        comparison_results = {
            "models": list(model_results.keys()),
            "model_stats": {},
            "variant_comparison": {},
            "overall_ranking": {}
        }
        
        # Analyze each model
        for model_name, results in model_results.items():
            print(f"Analyzing model: {model_name}")
            
            model_stats = self._analyze_model_performance(results)
            comparison_results["model_stats"][model_name] = model_stats
        
        # Compare across style variants
        comparison_results["variant_comparison"] = self._compare_by_variants(
            model_results
        )
        
        # Create overall ranking
        comparison_results["overall_ranking"] = self._rank_models(
            comparison_results["model_stats"]
        )
        
        # Save comparison results
        comparison_output = f"{output_dir}/model_comparison.json"
        DataUtils.save_json(comparison_results, comparison_output)
        
        # Generate comparison report
        self._generate_comparison_report(comparison_results, output_dir)
        
        print(f"Model comparison completed. Results saved to: {output_dir}")
        return comparison_results
    
    def _analyze_model_performance(self, results: DatasetDict) -> Dict[str, Any]:
        """
        Analyze performance of a single QA model.
        
        Args:
            results: QA results for the model
            
        Returns:
            Performance analysis
        """
        analysis = {
            "total_samples": 0,
            "style_variants": set(),
            "split_stats": {},
            "variant_stats": {},
            "overall_stats": {}
        }
        
        all_samples = []
        
        # Analyze each split
        for split_name, split_data in results.items():
            split_samples = []
            split_variants = set()
            
            for sample in split_data:
                sample_data = {
                    "style_variant": sample.get("style_variant", "unknown"),
                    "has_answer": len(sample.get("generated_answer", "").strip()) > 0,
                    "answer_length": len(sample.get("generated_answer", "")),
                    "question_length": len(sample.get("modified_question", "")),
                }
                
                split_samples.append(sample_data)
                split_variants.add(sample_data["style_variant"])
            
            # Calculate split statistics
            split_stats = self._calculate_split_stats(split_samples)
            analysis["split_stats"][split_name] = split_stats
            
            all_samples.extend(split_samples)
            analysis["style_variants"].update(split_variants)
        
        # Overall statistics
        analysis["total_samples"] = len(all_samples)
        analysis["style_variants"] = list(analysis["style_variants"])
        analysis["overall_stats"] = self._calculate_split_stats(all_samples)
        
        # Variant-specific statistics
        for variant in analysis["style_variants"]:
            variant_samples = [s for s in all_samples if s["style_variant"] == variant]
            if variant_samples:
                analysis["variant_stats"][variant] = self._calculate_split_stats(variant_samples)
        
        return analysis
    
    def _calculate_split_stats(self, samples: List[Dict]) -> Dict[str, Any]:
        """
        Calculate statistics for a set of samples.
        
        Args:
            samples: List of sample data dictionaries
            
        Returns:
            Statistical summary
        """
        if not samples:
            return {
                "total_samples": 0,
                "answer_rate": 0.0,
                "avg_answer_length": 0.0,
                "avg_question_length": 0.0,
                "answer_length_std": 0.0
            }
        
        total_samples = len(samples)
        answers_generated = sum(1 for s in samples if s["has_answer"])
        answer_lengths = [s["answer_length"] for s in samples if s["has_answer"]]
        question_lengths = [s["question_length"] for s in samples]
        
        return {
            "total_samples": total_samples,
            "answer_rate": answers_generated / total_samples,
            "avg_answer_length": statistics.mean(answer_lengths) if answer_lengths else 0.0,
            "avg_question_length": statistics.mean(question_lengths) if question_lengths else 0.0,
            "answer_length_std": statistics.stdev(answer_lengths) if len(answer_lengths) > 1 else 0.0,
            "question_length_std": statistics.stdev(question_lengths) if len(question_lengths) > 1 else 0.0,
        }
    
    def _compare_by_variants(
        self, 
        model_results: Dict[str, DatasetDict]
    ) -> Dict[str, Any]:
        """
        Compare model performance across style variants.
        
        Args:
            model_results: Model results to compare
            
        Returns:
            Variant-based comparison
        """
        # Get all unique variants across models
        all_variants = set()
        for results in model_results.values():
            for split_data in results.values():
                for sample in split_data:
                    all_variants.add(sample.get("style_variant", "unknown"))
        
        all_variants = list(all_variants)
        
        variant_comparison = {
            "variants": all_variants,
            "model_performance": {}
        }
        
        # Compare each model's performance on each variant
        for model_name, results in model_results.items():
            model_variant_stats = {}
            
            for variant in all_variants:
                # Collect all samples for this variant
                variant_samples = []
                for split_data in results.values():
                    for sample in split_data:
                        if sample.get("style_variant") == variant:
                            variant_samples.append({
                                "has_answer": len(sample.get("generated_answer", "").strip()) > 0,
                                "answer_length": len(sample.get("generated_answer", "")),
                                "question_length": len(sample.get("modified_question", "")),
                            })
                
                if variant_samples:
                    model_variant_stats[variant] = self._calculate_split_stats(variant_samples)
                else:
                    model_variant_stats[variant] = self._calculate_split_stats([])
            
            variant_comparison["model_performance"][model_name] = model_variant_stats
        
        return variant_comparison
    
    def _rank_models(self, model_stats: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Create overall ranking of models based on performance metrics.
        
        Args:
            model_stats: Statistics for each model
            
        Returns:
            Model ranking
        """
        ranking_metrics = ["answer_rate", "avg_answer_length"]
        model_scores = {}
        
        for model_name, stats in model_stats.items():
            overall_stats = stats.get("overall_stats", {})
            
            # Calculate composite score (weighted average)
            score = (
                overall_stats.get("answer_rate", 0) * 0.7 +  # 70% weight on answer rate
                min(1.0, overall_stats.get("avg_answer_length", 0) / 200) * 0.3  # 30% weight on answer length (normalized)
            )
            
            model_scores[model_name] = {
                "composite_score": score,
                "answer_rate": overall_stats.get("answer_rate", 0),
                "avg_answer_length": overall_stats.get("avg_answer_length", 0),
                "total_samples": stats.get("total_samples", 0)
            }
        
        # Sort by composite score
        ranked_models = sorted(
            model_scores.items(), 
            key=lambda x: x[1]["composite_score"], 
            reverse=True
        )
        
        return {
            "ranking": [model_name for model_name, _ in ranked_models],
            "scores": model_scores
        }
    
    def _generate_comparison_report(
        self, 
        comparison_results: Dict[str, Any], 
        output_dir: str
    ) -> None:
        """
        Generate a human-readable comparison report.
        
        Args:
            comparison_results: Results from model comparison
            output_dir: Output directory for report
        """
        report_lines = []
        
        # Title
        report_lines.append("# SPQA Model Comparison Report\n")
        
        # Models compared
        models = comparison_results["models"]
        report_lines.append(f"## Models Compared: {len(models)}")
        for i, model in enumerate(models, 1):
            report_lines.append(f"{i}. {model}")
        report_lines.append("")
        
        # Overall ranking
        ranking = comparison_results["overall_ranking"]
        report_lines.append("## Overall Performance Ranking")
        for i, model_name in enumerate(ranking["ranking"], 1):
            score_info = ranking["scores"][model_name]
            report_lines.append(
                f"{i}. **{model_name}** - "
                f"Score: {score_info['composite_score']:.3f}, "
                f"Answer Rate: {score_info['answer_rate']:.1%}, "
                f"Avg Length: {score_info['avg_answer_length']:.1f} chars"
            )
        report_lines.append("")
        
        # Per-model details
        report_lines.append("## Model Details")
        model_stats = comparison_results["model_stats"]
        
        for model_name in models:
            stats = model_stats[model_name]
            report_lines.append(f"### {model_name}")
            report_lines.append(f"- Total Samples: {stats['total_samples']:,}")
            report_lines.append(f"- Style Variants: {len(stats['style_variants'])}")
            report_lines.append(f"- Overall Answer Rate: {stats['overall_stats']['answer_rate']:.1%}")
            report_lines.append(f"- Avg Answer Length: {stats['overall_stats']['avg_answer_length']:.1f} chars")
            report_lines.append("")
        
        # Style variant comparison
        variant_comp = comparison_results["variant_comparison"]
        report_lines.append("## Performance by Style Variant")
        
        for variant in variant_comp["variants"]:
            report_lines.append(f"### {variant.title()} Style")
            
            variant_scores = []
            for model_name in models:
                model_variant_stats = variant_comp["model_performance"][model_name]
                if variant in model_variant_stats:
                    answer_rate = model_variant_stats[variant]["answer_rate"]
                    variant_scores.append((model_name, answer_rate))
            
            # Sort by answer rate for this variant
            variant_scores.sort(key=lambda x: x[1], reverse=True)
            
            for i, (model_name, answer_rate) in enumerate(variant_scores, 1):
                report_lines.append(f"{i}. {model_name}: {answer_rate:.1%}")
            
            report_lines.append("")
        
        # Save report
        report_content = "\n".join(report_lines)
        report_path = f"{output_dir}/comparison_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Comparison report saved to: {report_path}")
    
    def analyze_style_robustness(
        self,
        qa_results: DatasetDict,
        baseline_variant: str = "original"
    ) -> Dict[str, Any]:
        """
        Analyze model robustness to style perturbations.
        
        Args:
            qa_results: QA results dataset
            baseline_variant: Baseline style variant for comparison
            
        Returns:
            Style robustness analysis
        """
        print("Analyzing style robustness...")
        
        analysis = {
            "baseline_variant": baseline_variant,
            "style_variants": [],
            "robustness_metrics": {},
            "degradation_analysis": {}
        }
        
        # Collect all variants and baseline performance
        all_variants = set()
        baseline_samples = []
        variant_samples = {}
        
        for split_data in qa_results.values():
            for sample in split_data:
                variant = sample.get("style_variant", "unknown")
                all_variants.add(variant)
                
                sample_metrics = {
                    "serial_number": sample.get("serial_number", -1),
                    "has_answer": len(sample.get("generated_answer", "").strip()) > 0,
                    "answer_length": len(sample.get("generated_answer", "")),
                    "question_length": len(sample.get("modified_question", "")),
                }
                
                if variant == baseline_variant:
                    baseline_samples.append(sample_metrics)
                else:
                    if variant not in variant_samples:
                        variant_samples[variant] = []
                    variant_samples[variant].append(sample_metrics)
        
        analysis["style_variants"] = list(all_variants)
        
        # Calculate baseline performance
        baseline_stats = self._calculate_split_stats(baseline_samples)
        analysis["baseline_performance"] = baseline_stats
        
        # Compare each variant to baseline
        for variant, samples in variant_samples.items():
            variant_stats = self._calculate_split_stats(samples)
            
            # Calculate degradation metrics
            answer_rate_change = variant_stats["answer_rate"] - baseline_stats["answer_rate"]
            length_change = variant_stats["avg_answer_length"] - baseline_stats["avg_answer_length"]
            
            degradation = {
                "variant_performance": variant_stats,
                "answer_rate_change": answer_rate_change,
                "answer_rate_change_pct": answer_rate_change / baseline_stats["answer_rate"] if baseline_stats["answer_rate"] > 0 else 0,
                "avg_length_change": length_change,
                "avg_length_change_pct": length_change / baseline_stats["avg_answer_length"] if baseline_stats["avg_answer_length"] > 0 else 0,
                "robustness_score": 1.0 - abs(answer_rate_change)  # Simple robustness score
            }
            
            analysis["degradation_analysis"][variant] = degradation
        
        # Calculate overall robustness metrics
        robustness_scores = [
            deg["robustness_score"] for deg in analysis["degradation_analysis"].values()
        ]
        
        analysis["robustness_metrics"] = {
            "avg_robustness_score": statistics.mean(robustness_scores) if robustness_scores else 0,
            "min_robustness_score": min(robustness_scores) if robustness_scores else 0,
            "max_robustness_score": max(robustness_scores) if robustness_scores else 0,
            "robustness_std": statistics.stdev(robustness_scores) if len(robustness_scores) > 1 else 0,
            "most_robust_variant": max(analysis["degradation_analysis"].items(), key=lambda x: x[1]["robustness_score"])[0] if analysis["degradation_analysis"] else None,
            "least_robust_variant": min(analysis["degradation_analysis"].items(), key=lambda x: x[1]["robustness_score"])[0] if analysis["degradation_analysis"] else None,
        }
        
        return analysis
    
    def generate_performance_summary(
        self,
        qa_results: DatasetDict,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive performance summary.
        
        Args:
            qa_results: QA results dataset
            output_path: Path to save summary
            
        Returns:
            Performance summary
        """
        print("Generating performance summary...")
        
        # Basic analysis
        basic_analysis = self._analyze_model_performance(qa_results)
        
        # Robustness analysis  
        robustness_analysis = self.analyze_style_robustness(qa_results)
        
        # Combined summary
        summary = {
            "basic_performance": basic_analysis,
            "style_robustness": robustness_analysis,
            "key_insights": self._generate_key_insights(basic_analysis, robustness_analysis)
        }
        
        # Save summary
        DataUtils.save_json(summary, output_path)
        
        print(f"Performance summary saved to: {output_path}")
        return summary
    
    def _generate_key_insights(
        self, 
        basic_analysis: Dict, 
        robustness_analysis: Dict
    ) -> List[str]:
        """
        Generate key insights from the analysis.
        
        Args:
            basic_analysis: Basic performance analysis
            robustness_analysis: Style robustness analysis
            
        Returns:
            List of key insights
        """
        insights = []
        
        # Basic performance insights
        overall_stats = basic_analysis["overall_stats"]
        answer_rate = overall_stats["answer_rate"]
        
        if answer_rate > 0.9:
            insights.append("High answer generation rate (>90%)")
        elif answer_rate < 0.5:
            insights.append("Low answer generation rate (<50%) - potential model issues")
        
        # Style robustness insights
        robustness_metrics = robustness_analysis["robustness_metrics"]
        avg_robustness = robustness_metrics["avg_robustness_score"]
        
        if avg_robustness > 0.8:
            insights.append("Good robustness to style perturbations")
        elif avg_robustness < 0.6:
            insights.append("Poor robustness to style perturbations - significant performance degradation")
        
        # Variant-specific insights
        least_robust = robustness_metrics.get("least_robust_variant")
        if least_robust:
            insights.append(f"Most challenging style variant: {least_robust}")
        
        most_robust = robustness_metrics.get("most_robust_variant")
        if most_robust:
            insights.append(f"Most robust style variant: {most_robust}")
        
        return insights
