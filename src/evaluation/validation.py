"""
Judge validation utilities for SPQA framework.

This module provides utilities for validating the LLM-Judge against human annotations
and ensuring the reliability of automated evaluation.
"""

import statistics
from typing import Dict, List, Any, Optional, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
import numpy as np

from ..utils.config import Config
from ..utils.data_utils import DataUtils


class JudgeValidator:
    """
    Validator for LLM-Judge reliability assessment.
    
    This class implements validation methods to ensure that the automated
    LLM-Judge evaluations align well with human expert judgments.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize JudgeValidator.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or Config()
        self.criteria = ["correctness", "completeness", "coherence", "linguistic_adaptability"]
    
    def validate_against_human_annotations(
        self,
        llm_judge_results: List[Dict[str, Any]],
        human_annotations: List[Dict[str, Any]],
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Validate LLM-Judge results against human annotations.
        
        Args:
            llm_judge_results: Results from LLM-Judge evaluation
            human_annotations: Human expert annotations  
            output_dir: Output directory for validation results
            
        Returns:
            Comprehensive validation analysis
        """
        print("Validating LLM-Judge against human annotations...")
        
        validation_results = {
            "validation_config": {
                "total_llm_samples": len(llm_judge_results),
                "total_human_samples": len(human_annotations),
                "criteria": self.criteria
            },
            "alignment_analysis": {},
            "reliability_metrics": {},
            "bias_analysis": {},
            "recommendation": ""
        }
        
        # Align samples between LLM-Judge and human annotations
        aligned_samples = self._align_samples(llm_judge_results, human_annotations)
        
        if not aligned_samples:
            validation_results["error"] = "No aligned samples found between LLM-Judge and human annotations"
            return validation_results
        
        validation_results["validation_config"]["aligned_samples"] = len(aligned_samples)
        
        # Calculate alignment metrics
        validation_results["alignment_analysis"] = self._calculate_alignment_metrics(aligned_samples)
        
        # Calculate reliability metrics
        validation_results["reliability_metrics"] = self._calculate_reliability_metrics(aligned_samples)
        
        # Analyze bias patterns
        validation_results["bias_analysis"] = self._analyze_bias_patterns(aligned_samples)
        
        # Generate recommendation
        validation_results["recommendation"] = self._generate_validation_recommendation(
            validation_results["alignment_analysis"],
            validation_results["reliability_metrics"]
        )
        
        # Save validation results
        validation_output = f"{output_dir}/judge_validation.json"
        DataUtils.save_json(validation_results, validation_output)
        
        # Generate validation report
        self._generate_validation_report(validation_results, f"{output_dir}/validation_report.md")
        
        print(f"Judge validation completed. Results saved to: {output_dir}")
        return validation_results
    
    def _align_samples(
        self,
        llm_judge_results: List[Dict[str, Any]],
        human_annotations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Align samples between LLM-Judge results and human annotations.
        
        Args:
            llm_judge_results: LLM-Judge evaluation results
            human_annotations: Human expert annotations
            
        Returns:
            List of aligned sample pairs
        """
        aligned_samples = []
        
        # Create lookup for human annotations by serial number
        human_lookup = {}
        for human_sample in human_annotations:
            serial_num = human_sample.get("serial_number")
            if serial_num is not None:
                human_lookup[serial_num] = human_sample
        
        # Align LLM-Judge results with human annotations
        for llm_sample in llm_judge_results:
            serial_num = llm_sample.get("serial_number")
            if serial_num in human_lookup:
                human_sample = human_lookup[serial_num]
                
                # Ensure both samples have valid scores
                llm_scores = llm_sample.get("scores", {})
                human_scores = human_sample.get("scores", {})
                
                if (isinstance(llm_scores, dict) and isinstance(human_scores, dict) and
                    all(criterion in llm_scores for criterion in self.criteria) and
                    all(criterion in human_scores for criterion in self.criteria)):
                    
                    aligned_samples.append({
                        "serial_number": serial_num,
                        "llm_scores": llm_scores,
                        "human_scores": human_scores,
                        "style_variant": llm_sample.get("style_variant", "unknown"),
                        "question": llm_sample.get("modified_question", ""),
                        "answer": llm_sample.get("generated_answer", "")
                    })
        
        return aligned_samples
    
    def _calculate_alignment_metrics(self, aligned_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate alignment metrics between LLM-Judge and human annotations.
        
        Args:
            aligned_samples: List of aligned sample pairs
            
        Returns:
            Alignment analysis results
        """
        alignment_metrics = {
            "criterion_correlations": {},
            "criterion_agreements": {},
            "overall_correlation": 0.0,
            "overall_agreement": 0.0
        }
        
        # Calculate per-criterion metrics
        all_llm_scores = []
        all_human_scores = []
        
        for criterion in self.criteria:
            llm_scores = [sample["llm_scores"][criterion] for sample in aligned_samples]
            human_scores = [sample["human_scores"][criterion] for sample in aligned_samples]
            
            # Pearson correlation
            try:
                correlation, p_value = pearsonr(llm_scores, human_scores)
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0
                p_value = 1.0
            
            # Cohen's Kappa (treating as categorical)
            try:
                kappa = cohen_kappa_score(human_scores, llm_scores)
                if np.isnan(kappa):
                    kappa = 0.0
            except:
                kappa = 0.0
            
            # Exact agreement percentage
            exact_matches = sum(1 for h, l in zip(human_scores, llm_scores) if h == l)
            exact_agreement = exact_matches / len(aligned_samples) * 100
            
            # Within-1-point agreement
            within_1_matches = sum(1 for h, l in zip(human_scores, llm_scores) if abs(h - l) <= 1)
            within_1_agreement = within_1_matches / len(aligned_samples) * 100
            
            alignment_metrics["criterion_correlations"][criterion] = {
                "pearson_r": round(correlation, 3),
                "p_value": round(p_value, 4),
                "strength": self._interpret_correlation(abs(correlation))
            }
            
            alignment_metrics["criterion_agreements"][criterion] = {
                "cohen_kappa": round(kappa, 3),
                "exact_agreement_pct": round(exact_agreement, 1),
                "within_1_agreement_pct": round(within_1_agreement, 1),
                "kappa_interpretation": self._interpret_kappa(kappa)
            }
            
            # Add to overall scores
            all_llm_scores.extend(llm_scores)
            all_human_scores.extend(human_scores)
        
        # Overall alignment metrics
        try:
            overall_correlation, _ = pearsonr(all_llm_scores, all_human_scores)
            if np.isnan(overall_correlation):
                overall_correlation = 0.0
        except:
            overall_correlation = 0.0
        
        try:
            overall_kappa = cohen_kappa_score(all_human_scores, all_llm_scores)
            if np.isnan(overall_kappa):
                overall_kappa = 0.0
        except:
            overall_kappa = 0.0
        
        alignment_metrics["overall_correlation"] = round(overall_correlation, 3)
        alignment_metrics["overall_agreement"] = round(overall_kappa, 3)
        
        return alignment_metrics
    
    def _calculate_reliability_metrics(self, aligned_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate reliability metrics for the LLM-Judge.
        
        Args:
            aligned_samples: List of aligned sample pairs
            
        Returns:
            Reliability analysis results
        """
        reliability_metrics = {
            "inter_rater_reliability": {},
            "systematic_bias": {},
            "variance_analysis": {}
        }
        
        # Calculate inter-rater reliability for each criterion
        for criterion in self.criteria:
            llm_scores = [sample["llm_scores"][criterion] for sample in aligned_samples]
            human_scores = [sample["human_scores"][criterion] for sample in aligned_samples]
            
            # Mean absolute error
            mae = statistics.mean([abs(h - l) for h, l in zip(human_scores, llm_scores)])
            
            # Root mean square error
            rmse = statistics.sqrt(statistics.mean([(h - l)**2 for h, l in zip(human_scores, llm_scores)]))
            
            # Bias (systematic over/under-scoring)
            bias = statistics.mean([l - h for h, l in zip(human_scores, llm_scores)])
            
            reliability_metrics["inter_rater_reliability"][criterion] = {
                "mae": round(mae, 3),
                "rmse": round(rmse, 3)
            }
            
            reliability_metrics["systematic_bias"][criterion] = {
                "bias": round(bias, 3),
                "direction": "over-scoring" if bias > 0.1 else "under-scoring" if bias < -0.1 else "balanced"
            }
            
            # Variance analysis
            llm_variance = statistics.variance(llm_scores) if len(llm_scores) > 1 else 0
            human_variance = statistics.variance(human_scores) if len(human_scores) > 1 else 0
            
            reliability_metrics["variance_analysis"][criterion] = {
                "llm_variance": round(llm_variance, 3),
                "human_variance": round(human_variance, 3),
                "variance_ratio": round(llm_variance / human_variance, 3) if human_variance > 0 else 0
            }
        
        return reliability_metrics
    
    def _analyze_bias_patterns(self, aligned_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze bias patterns in LLM-Judge evaluations.
        
        Args:
            aligned_samples: List of aligned sample pairs
            
        Returns:
            Bias pattern analysis
        """
        bias_analysis = {
            "style_variant_bias": {},
            "score_level_bias": {},
            "consistency_analysis": {}
        }
        
        # Group by style variant
        variant_groups = {}
        for sample in aligned_samples:
            variant = sample["style_variant"]
            if variant not in variant_groups:
                variant_groups[variant] = []
            variant_groups[variant].append(sample)
        
        # Analyze bias by style variant
        for variant, samples in variant_groups.items():
            if len(samples) < 3:  # Skip variants with too few samples
                continue
            
            variant_bias = {}
            for criterion in self.criteria:
                llm_scores = [s["llm_scores"][criterion] for s in samples]
                human_scores = [s["human_scores"][criterion] for s in samples]
                
                bias = statistics.mean([l - h for h, l in zip(human_scores, llm_scores)])
                variant_bias[criterion] = round(bias, 3)
            
            bias_analysis["style_variant_bias"][variant] = variant_bias
        
        # Analyze bias by score level
        for criterion in self.criteria:
            score_level_bias = {1: [], 2: [], 3: []}
            
            for sample in aligned_samples:
                human_score = sample["human_scores"][criterion]
                llm_score = sample["llm_scores"][criterion]
                bias = llm_score - human_score
                
                if human_score in score_level_bias:
                    score_level_bias[human_score].append(bias)
            
            # Calculate average bias for each score level
            level_bias = {}
            for level, biases in score_level_bias.items():
                if biases:
                    level_bias[f"score_{level}"] = round(statistics.mean(biases), 3)
                else:
                    level_bias[f"score_{level}"] = 0.0
            
            bias_analysis["score_level_bias"][criterion] = level_bias
        
        # Consistency analysis
        bias_analysis["consistency_analysis"] = self._analyze_consistency(aligned_samples)
        
        return bias_analysis
    
    def _analyze_consistency(self, aligned_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze scoring consistency of the LLM-Judge.
        
        Args:
            aligned_samples: List of aligned sample pairs
            
        Returns:
            Consistency analysis results
        """
        consistency_metrics = {}
        
        for criterion in self.criteria:
            # Calculate disagreement patterns
            disagreements = []
            for sample in aligned_samples:
                human_score = sample["human_scores"][criterion]
                llm_score = sample["llm_scores"][criterion]
                disagreements.append(abs(human_score - llm_score))
            
            # Consistency metrics
            avg_disagreement = statistics.mean(disagreements)
            max_disagreement = max(disagreements)
            consistency_rate = sum(1 for d in disagreements if d <= 1) / len(disagreements) * 100
            
            consistency_metrics[criterion] = {
                "avg_disagreement": round(avg_disagreement, 3),
                "max_disagreement": max_disagreement,
                "within_1_consistency_pct": round(consistency_rate, 1)
            }
        
        return consistency_metrics
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength."""
        if correlation >= 0.7:
            return "strong"
        elif correlation >= 0.5:
            return "moderate"
        elif correlation >= 0.3:
            return "weak"
        else:
            return "very weak"
    
    def _interpret_kappa(self, kappa: float) -> str:
        """Interpret Cohen's Kappa value."""
        if kappa >= 0.8:
            return "excellent"
        elif kappa >= 0.6:
            return "substantial"
        elif kappa >= 0.4:
            return "moderate"
        elif kappa >= 0.2:
            return "fair"
        else:
            return "poor"
    
    def _generate_validation_recommendation(
        self,
        alignment_analysis: Dict[str, Any],
        reliability_metrics: Dict[str, Any]
    ) -> str:
        """
        Generate recommendation based on validation results.
        
        Args:
            alignment_analysis: Alignment analysis results
            reliability_metrics: Reliability metrics
            
        Returns:
            Recommendation string
        """
        overall_correlation = alignment_analysis["overall_correlation"]
        overall_agreement = alignment_analysis["overall_agreement"]
        
        if overall_correlation >= 0.6 and overall_agreement >= 0.4:
            return "LLM-Judge shows good alignment with human annotations and can be used reliably for automated evaluation."
        elif overall_correlation >= 0.4 and overall_agreement >= 0.3:
            return "LLM-Judge shows moderate alignment with human annotations. Consider validation on additional samples or prompt refinement."
        elif overall_correlation >= 0.3:
            return "LLM-Judge shows weak alignment with human annotations. Significant improvements needed before deployment."
        else:
            return "LLM-Judge shows poor alignment with human annotations. Not recommended for automated evaluation without major improvements."
    
    def _generate_validation_report(self, validation_results: Dict[str, Any], output_path: str) -> None:
        """
        Generate a markdown validation report.
        
        Args:
            validation_results: Validation analysis results
            output_path: Output path for the report
        """
        lines = []
        
        lines.append("# LLM-Judge Validation Report")
        lines.append("")
        
        # Summary
        config = validation_results["validation_config"]
        lines.append("## Validation Summary")
        lines.append(f"- **Total LLM-Judge Samples**: {config['total_llm_samples']:,}")
        lines.append(f"- **Total Human Annotations**: {config['total_human_samples']:,}")
        lines.append(f"- **Aligned Samples**: {config.get('aligned_samples', 0):,}")
        lines.append("")
        
        # Overall metrics
        alignment = validation_results["alignment_analysis"]
        lines.append("## Overall Alignment")
        lines.append(f"- **Overall Correlation**: {alignment['overall_correlation']:.3f}")
        lines.append(f"- **Overall Agreement (Kappa)**: {alignment['overall_agreement']:.3f}")
        lines.append("")
        
        # Per-criterion results
        lines.append("## Per-Criterion Analysis")
        for criterion in self.criteria:
            if criterion in alignment["criterion_correlations"]:
                corr_data = alignment["criterion_correlations"][criterion]
                agree_data = alignment["criterion_agreements"][criterion]
                
                lines.append(f"### {criterion.replace('_', ' ').title()}")
                lines.append(f"- **Correlation**: {corr_data['pearson_r']} ({corr_data['strength']})")
                lines.append(f"- **Kappa**: {agree_data['cohen_kappa']} ({agree_data['kappa_interpretation']})")
                lines.append(f"- **Exact Agreement**: {agree_data['exact_agreement_pct']}%")
                lines.append("")
        
        # Recommendation
        lines.append("## Recommendation")
        lines.append(validation_results["recommendation"])
        lines.append("")
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"Validation report saved to: {output_path}")
    
    def calculate_inter_annotator_agreement(
        self,
        annotations1: List[Dict[str, Any]],
        annotations2: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate inter-annotator agreement between two sets of human annotations.
        
        Args:
            annotations1: First annotator's evaluations
            annotations2: Second annotator's evaluations
            
        Returns:
            Inter-annotator agreement metrics
        """
        # Align the two annotation sets
        aligned_pairs = []
        
        # Create lookup for second annotator
        lookup2 = {ann.get("serial_number"): ann for ann in annotations2}
        
        for ann1 in annotations1:
            serial_num = ann1.get("serial_number")
            if serial_num in lookup2:
                ann2 = lookup2[serial_num]
                
                scores1 = ann1.get("scores", {})
                scores2 = ann2.get("scores", {})
                
                if (isinstance(scores1, dict) and isinstance(scores2, dict) and
                    all(criterion in scores1 for criterion in self.criteria) and
                    all(criterion in scores2 for criterion in self.criteria)):
                    
                    aligned_pairs.append({
                        "serial_number": serial_num,
                        "scores1": scores1,
                        "scores2": scores2
                    })
        
        if not aligned_pairs:
            return {"error": "No aligned annotations found"}
        
        # Calculate agreement metrics
        agreement_metrics = {
            "total_aligned": len(aligned_pairs),
            "criterion_agreements": {}
        }
        
        for criterion in self.criteria:
            scores1 = [pair["scores1"][criterion] for pair in aligned_pairs]
            scores2 = [pair["scores2"][criterion] for pair in aligned_pairs]
            
            # Pearson correlation
            try:
                correlation, p_value = pearsonr(scores1, scores2)
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0
                p_value = 1.0
            
            # Cohen's Kappa
            try:
                kappa = cohen_kappa_score(scores1, scores2)
                if np.isnan(kappa):
                    kappa = 0.0
            except:
                kappa = 0.0
            
            # Exact agreement
            exact_matches = sum(1 for s1, s2 in zip(scores1, scores2) if s1 == s2)
            exact_agreement = exact_matches / len(aligned_pairs) * 100
            
            agreement_metrics["criterion_agreements"][criterion] = {
                "pearson_r": round(correlation, 3),
                "p_value": round(p_value, 4),
                "cohen_kappa": round(kappa, 3),
                "exact_agreement_pct": round(exact_agreement, 1),
                "kappa_interpretation": self._interpret_kappa(kappa)
            }
        
        return agreement_metrics
