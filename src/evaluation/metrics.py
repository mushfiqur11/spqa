"""
Evaluation metrics for SPQA framework.

This module provides metrics calculation and statistical analysis
for the LLM-Judge evaluation results.
"""

import statistics
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class EvaluationMetrics:
    """
    Metrics calculation and analysis for SPQA evaluation results.
    
    This class handles the calculation of comprehensive metrics from
    LLM-Judge evaluation scores across the four criteria.
    """
    
    def __init__(self):
        """Initialize evaluation metrics calculator."""
        self.criteria = ["correctness", "completeness", "coherence", "linguistic_adaptability"]
    
    def calculate_overall_metrics(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate overall metrics across all evaluations.
        
        Args:
            evaluations: List of evaluation results with scores
            
        Returns:
            Dictionary of overall metrics
        """
        if not evaluations:
            return self._get_empty_metrics()
        
        # Extract scores for each criterion
        criterion_scores = {criterion: [] for criterion in self.criteria}
        valid_evaluations = []
        
        for evaluation in evaluations:
            if "scores" in evaluation and isinstance(evaluation["scores"], dict):
                valid_evaluations.append(evaluation)
                for criterion in self.criteria:
                    score = evaluation["scores"].get(criterion)
                    if isinstance(score, (int, float)) and 1 <= score <= 3:
                        criterion_scores[criterion].append(score)
        
        if not valid_evaluations:
            return self._get_empty_metrics()
        
        # Calculate basic statistics for each criterion
        metrics = {
            "total_samples": len(evaluations),
            "valid_evaluations": len(valid_evaluations),
            "evaluation_success_rate": len(valid_evaluations) / len(evaluations)
        }
        
        # Calculate metrics for each criterion
        for criterion in self.criteria:
            scores = criterion_scores[criterion]
            if scores:
                metrics.update({
                    f"avg_{criterion}": statistics.mean(scores),
                    f"std_{criterion}": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    f"min_{criterion}": min(scores),
                    f"max_{criterion}": max(scores),
                    f"median_{criterion}": statistics.median(scores)
                })
                
                # Score distribution
                metrics[f"{criterion}_distribution"] = {
                    "score_1": scores.count(1),
                    "score_2": scores.count(2),
                    "score_3": scores.count(3),
                    "score_1_pct": scores.count(1) / len(scores) * 100,
                    "score_2_pct": scores.count(2) / len(scores) * 100,
                    "score_3_pct": scores.count(3) / len(scores) * 100
                }
            else:
                # Default values when no valid scores
                metrics.update({
                    f"avg_{criterion}": 0.0,
                    f"std_{criterion}": 0.0,
                    f"min_{criterion}": 0,
                    f"max_{criterion}": 0,
                    f"median_{criterion}": 0.0,
                    f"{criterion}_distribution": {
                        "score_1": 0, "score_2": 0, "score_3": 0,
                        "score_1_pct": 0.0, "score_2_pct": 0.0, "score_3_pct": 0.0
                    }
                })
        
        # Calculate composite metrics
        metrics.update(self._calculate_composite_metrics(criterion_scores))
        
        return metrics
    
    def calculate_split_metrics(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate metrics for a specific data split.
        
        Args:
            evaluations: List of evaluation results for the split
            
        Returns:
            Dictionary of split-specific metrics
        """
        # Use the same calculation as overall metrics
        split_metrics = self.calculate_overall_metrics(evaluations)
        
        # Add split-specific analysis
        if evaluations:
            # Analyze by style variants within the split
            variant_analysis = self._analyze_by_style_variants(evaluations)
            split_metrics["style_variant_analysis"] = variant_analysis
            
            # Calculate degradation metrics if original variant exists
            degradation_analysis = self._calculate_degradation_metrics(evaluations)
            if degradation_analysis:
                split_metrics["degradation_analysis"] = degradation_analysis
        
        return split_metrics
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """
        Get empty metrics structure for when no evaluations are available.
        
        Returns:
            Empty metrics dictionary
        """
        empty_metrics = {
            "total_samples": 0,
            "valid_evaluations": 0,
            "evaluation_success_rate": 0.0
        }
        
        for criterion in self.criteria:
            empty_metrics.update({
                f"avg_{criterion}": 0.0,
                f"std_{criterion}": 0.0,
                f"min_{criterion}": 0,
                f"max_{criterion}": 0,
                f"median_{criterion}": 0.0,
                f"{criterion}_distribution": {
                    "score_1": 0, "score_2": 0, "score_3": 0,
                    "score_1_pct": 0.0, "score_2_pct": 0.0, "score_3_pct": 0.0
                }
            })
        
        return empty_metrics
    
    def _calculate_composite_metrics(self, criterion_scores: Dict[str, List]) -> Dict[str, Any]:
        """
        Calculate composite metrics across all criteria.
        
        Args:
            criterion_scores: Dictionary mapping criteria to score lists
            
        Returns:
            Dictionary of composite metrics
        """
        composite_metrics = {}
        
        # Calculate weighted average performance
        all_scores = []
        weights = {
            "correctness": 0.3,
            "completeness": 0.3,
            "coherence": 0.2,
            "linguistic_adaptability": 0.2
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for criterion, weight in weights.items():
            scores = criterion_scores.get(criterion, [])
            if scores:
                avg_score = statistics.mean(scores)
                weighted_sum += avg_score * weight
                total_weight += weight
                all_scores.extend(scores)
        
        composite_metrics["weighted_average_performance"] = (
            weighted_sum / total_weight if total_weight > 0 else 0.0
        )
        
        # Overall performance metrics
        if all_scores:
            composite_metrics["overall_average"] = statistics.mean(all_scores)
            composite_metrics["overall_std"] = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
            composite_metrics["performance_consistency"] = (
                1.0 - (composite_metrics["overall_std"] / composite_metrics["overall_average"])
                if composite_metrics["overall_average"] > 0 else 0.0
            )
        else:
            composite_metrics.update({
                "overall_average": 0.0,
                "overall_std": 0.0,
                "performance_consistency": 0.0
            })
        
        # Criterion correlation analysis
        composite_metrics["criterion_correlations"] = self._calculate_criterion_correlations(
            criterion_scores
        )
        
        return composite_metrics
    
    def _analyze_by_style_variants(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze performance by style variants.
        
        Args:
            evaluations: List of evaluation results
            
        Returns:
            Style variant analysis
        """
        variant_analysis = {}
        
        # Group by style variant
        variant_groups = {}
        for evaluation in evaluations:
            if "scores" not in evaluation or not isinstance(evaluation["scores"], dict):
                continue
            
            variant = evaluation.get("style_variant", "unknown")
            if variant not in variant_groups:
                variant_groups[variant] = []
            variant_groups[variant].append(evaluation)
        
        # Calculate metrics for each variant
        for variant, variant_evaluations in variant_groups.items():
            variant_metrics = self.calculate_overall_metrics(variant_evaluations)
            variant_analysis[variant] = {
                "total_samples": len(variant_evaluations),
                "metrics": variant_metrics
            }
        
        # Calculate variant comparison metrics
        if len(variant_groups) > 1:
            variant_comparison = self._compare_variants(variant_groups)
            variant_analysis["variant_comparison"] = variant_comparison
        
        return variant_analysis
    
    def _calculate_degradation_metrics(self, evaluations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Calculate performance degradation relative to original variant.
        
        Args:
            evaluations: List of evaluation results
            
        Returns:
            Degradation analysis or None if original variant not found
        """
        # Find original variant evaluations
        original_evaluations = [
            eval_item for eval_item in evaluations
            if eval_item.get("style_variant") == "original"
        ]
        
        if not original_evaluations:
            return None
        
        # Calculate baseline (original) performance
        baseline_metrics = self.calculate_overall_metrics(original_evaluations)
        
        # Calculate degradation for each variant
        degradation_analysis = {
            "baseline_variant": "original",
            "baseline_metrics": baseline_metrics,
            "variant_degradations": {}
        }
        
        # Group by variant
        variant_groups = {}
        for evaluation in evaluations:
            variant = evaluation.get("style_variant", "unknown")
            if variant != "original":
                if variant not in variant_groups:
                    variant_groups[variant] = []
                variant_groups[variant].append(evaluation)
        
        # Calculate degradation for each non-original variant
        for variant, variant_evaluations in variant_groups.items():
            variant_metrics = self.calculate_overall_metrics(variant_evaluations)
            
            # Calculate degradation for each criterion
            variant_degradation = {}
            for criterion in self.criteria:
                baseline_score = baseline_metrics.get(f"avg_{criterion}", 0)
                variant_score = variant_metrics.get(f"avg_{criterion}", 0)
                
                degradation = baseline_score - variant_score
                degradation_pct = (degradation / baseline_score * 100) if baseline_score > 0 else 0
                
                variant_degradation[criterion] = {
                    "baseline_score": baseline_score,
                    "variant_score": variant_score,
                    "absolute_degradation": degradation,
                    "relative_degradation_pct": degradation_pct
                }
            
            degradation_analysis["variant_degradations"][variant] = variant_degradation
        
        return degradation_analysis
    
    def _compare_variants(self, variant_groups: Dict[str, List]) -> Dict[str, Any]:
        """
        Compare performance across style variants.
        
        Args:
            variant_groups: Dictionary mapping variants to evaluation lists
            
        Returns:
            Variant comparison analysis
        """
        comparison = {
            "variant_rankings": {},
            "best_performing_variant": {},
            "worst_performing_variant": {},
            "variant_consistency": {}
        }
        
        # Calculate average scores for each variant and criterion
        variant_scores = {}
        for variant, evaluations in variant_groups.items():
            variant_metrics = self.calculate_overall_metrics(evaluations)
            variant_scores[variant] = {
                criterion: variant_metrics.get(f"avg_{criterion}", 0)
                for criterion in self.criteria
            }
        
        # Rank variants for each criterion
        for criterion in self.criteria:
            # Sort variants by performance on this criterion
            criterion_ranking = sorted(
                variant_scores.items(),
                key=lambda x: x[1][criterion],
                reverse=True
            )
            comparison["variant_rankings"][criterion] = [
                {"variant": variant, "score": scores[criterion]}
                for variant, scores in criterion_ranking
            ]
        
        # Identify best and worst performing variants overall
        overall_scores = {}
        for variant, scores in variant_scores.items():
            overall_scores[variant] = statistics.mean(scores.values())
        
        best_variant = max(overall_scores.items(), key=lambda x: x[1])
        worst_variant = min(overall_scores.items(), key=lambda x: x[1])
        
        comparison["best_performing_variant"] = {
            "variant": best_variant[0],
            "overall_score": best_variant[1]
        }
        comparison["worst_performing_variant"] = {
            "variant": worst_variant[0],
            "overall_score": worst_variant[1]
        }
        
        # Calculate consistency across variants
        for criterion in self.criteria:
            criterion_scores = [scores[criterion] for scores in variant_scores.values()]
            consistency = 1.0 - (statistics.stdev(criterion_scores) / statistics.mean(criterion_scores)) if statistics.mean(criterion_scores) > 0 else 0.0
            comparison["variant_consistency"][criterion] = consistency
        
        return comparison
    
    def _calculate_criterion_correlations(self, criterion_scores: Dict[str, List]) -> Dict[str, Any]:
        """
        Calculate correlations between evaluation criteria.
        
        Args:
            criterion_scores: Dictionary mapping criteria to score lists
            
        Returns:
            Correlation analysis
        """
        correlations = {}
        
        # Get aligned score vectors (same samples across criteria)
        aligned_scores = {}
        min_length = min(len(scores) for scores in criterion_scores.values() if scores)
        
        if min_length == 0:
            return {"error": "No aligned scores available for correlation analysis"}
        
        for criterion, scores in criterion_scores.items():
            if scores:
                aligned_scores[criterion] = scores[:min_length]
        
        # Calculate pairwise correlations
        criterion_list = list(aligned_scores.keys())
        correlation_matrix = {}
        
        for i, criterion1 in enumerate(criterion_list):
            correlation_matrix[criterion1] = {}
            for j, criterion2 in enumerate(criterion_list):
                if i == j:
                    correlation_matrix[criterion1][criterion2] = 1.0
                elif j < i:
                    # Use symmetry
                    correlation_matrix[criterion1][criterion2] = correlation_matrix[criterion2][criterion1]
                else:
                    # Calculate correlation
                    try:
                        scores1 = aligned_scores[criterion1]
                        scores2 = aligned_scores[criterion2]
                        
                        if len(scores1) > 1 and len(scores2) > 1:
                            corr = np.corrcoef(scores1, scores2)[0, 1]
                            if np.isnan(corr):
                                corr = 0.0
                        else:
                            corr = 0.0
                        
                        correlation_matrix[criterion1][criterion2] = round(corr, 3)
                    except:
                        correlation_matrix[criterion1][criterion2] = 0.0
        
        correlations["correlation_matrix"] = correlation_matrix
        correlations["aligned_samples"] = min_length
        
        return correlations
    
    def generate_performance_summary(self, metrics: Dict[str, Any]) -> str:
        """
        Generate a human-readable performance summary.
        
        Args:
            metrics: Calculated metrics dictionary
            
        Returns:
            Formatted performance summary
        """
        lines = ["# Performance Summary", ""]
        
        # Overall Statistics
        lines.append("## Overall Statistics")
        lines.append(f"- **Total Samples**: {metrics.get('total_samples', 0):,}")
        lines.append(f"- **Valid Evaluations**: {metrics.get('valid_evaluations', 0):,}")
        lines.append(f"- **Success Rate**: {metrics.get('evaluation_success_rate', 0):.1%}")
        lines.append("")
        
        # Criterion Performance
        lines.append("## Criterion Performance")
        for criterion in self.criteria:
            avg_score = metrics.get(f"avg_{criterion}", 0)
            std_score = metrics.get(f"std_{criterion}", 0)
            lines.append(f"- **{criterion.replace('_', ' ').title()}**: {avg_score:.2f} ± {std_score:.2f}")
        lines.append("")
        
        # Composite Metrics
        overall_avg = metrics.get("overall_average", 0)
        weighted_avg = metrics.get("weighted_average_performance", 0)
        consistency = metrics.get("performance_consistency", 0)
        
        lines.append("## Composite Performance")
        lines.append(f"- **Overall Average**: {overall_avg:.2f}/3.0")
        lines.append(f"- **Weighted Average**: {weighted_avg:.2f}/3.0")
        lines.append(f"- **Performance Consistency**: {consistency:.1%}")
        lines.append("")
        
        return "\n".join(lines)
    
    def export_metrics_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export a condensed metrics summary for reporting.
        
        Args:
            metrics: Full metrics dictionary
            
        Returns:
            Condensed metrics summary
        """
        summary = {
            "evaluation_overview": {
                "total_samples": metrics.get("total_samples", 0),
                "valid_evaluations": metrics.get("valid_evaluations", 0),
                "success_rate": metrics.get("evaluation_success_rate", 0)
            },
            "criterion_averages": {
                criterion: metrics.get(f"avg_{criterion}", 0)
                for criterion in self.criteria
            },
            "composite_scores": {
                "overall_average": metrics.get("overall_average", 0),
                "weighted_average": metrics.get("weighted_average_performance", 0),
                "consistency": metrics.get("performance_consistency", 0)
            },
            "performance_insights": self._generate_performance_insights(metrics)
        }
        
        return summary
    
    def _generate_performance_insights(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate key insights from metrics.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Analyze criterion performance
        criterion_scores = {
            criterion: metrics.get(f"avg_{criterion}", 0)
            for criterion in self.criteria
        }
        
        best_criterion = max(criterion_scores.items(), key=lambda x: x[1])
        worst_criterion = min(criterion_scores.items(), key=lambda x: x[1])
        
        insights.append(f"Strongest performance: {best_criterion[0].replace('_', ' ').title()} ({best_criterion[1]:.2f})")
        insights.append(f"Weakest performance: {worst_criterion[0].replace('_', ' ').title()} ({worst_criterion[1]:.2f})")
        
        # Overall performance assessment
        overall_avg = metrics.get("overall_average", 0)
        if overall_avg >= 2.5:
            insights.append("Excellent overall performance across all criteria")
        elif overall_avg >= 2.0:
            insights.append("Good overall performance with room for improvement")
        elif overall_avg >= 1.5:
            insights.append("Moderate performance - significant improvement needed")
        else:
            insights.append("Poor performance - major issues identified")
        
        # Consistency assessment
        consistency = metrics.get("performance_consistency", 0)
        if consistency >= 0.8:
            insights.append("High performance consistency across samples")
        elif consistency >= 0.6:
            insights.append("Moderate performance consistency")
        else:
            insights.append("Low performance consistency - results vary widely")
        
        return insights
