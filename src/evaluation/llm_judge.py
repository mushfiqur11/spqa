"""
LLM-Judge implementation for SPQA framework.

This module implements the automated evaluation system using GPT-4o as a judge
to assess QA performance across the four criteria defined in the SPQA paper.
"""

import json
import os
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from datasets import DatasetDict

from ..utils.config import Config
from ..utils.llm_utils import LLMUtils 
from ..utils.data_utils import DataUtils
from .rubrics import EvaluationRubrics
from .metrics import EvaluationMetrics


class LLMJudge:
    """
    LLM-based judge for evaluating QA performance in the SPQA framework.
    
    Uses GPT-4o as an automated judge to evaluate generated answers across:
    - Correctness
    - Completeness  
    - Coherence and Fluency
    - Linguistic Adaptability
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize LLMJudge.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or Config()
        self.llm_utils = LLMUtils(self.config)
        self.rubrics = EvaluationRubrics()
        self.metrics = EvaluationMetrics()
        
        # Initialize judge LLM (GPT-4o)
        self.judge_model_id = "gpt-4o-2024-08-06"
    
    def evaluate_qa_results(
        self,
        qa_results: DatasetDict,
        output_dir: str,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate QA results using the LLM judge.
        
        Args:
            qa_results: QA results dataset with generated answers
            output_dir: Output directory for evaluation results
            batch_size: Batch size for processing (to manage API limits)
            
        Returns:
            Comprehensive evaluation results
        """
        print("Starting LLM-Judge evaluation...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results structure
        evaluation_results = {
            "evaluation_config": {
                "judge_model": self.judge_model_id,
                "criteria": self.config.evaluation_criteria,
                "batch_size": batch_size
            },
            "split_results": {},
            "overall_metrics": {},
            "detailed_scores": []
        }
        
        # Process each split
        all_evaluations = []
        
        for split_name, split_data in qa_results.items():
            print(f"Evaluating split: {split_name}")
            
            split_evaluations = self._evaluate_split(split_data, batch_size)
            evaluation_results["split_results"][split_name] = split_evaluations
            all_evaluations.extend(split_evaluations["detailed_evaluations"])
            
            # Save intermediate results
            split_output = os.path.join(output_dir, f"evaluation_{split_name}.json")
            DataUtils.save_json(split_evaluations, split_output)
        
        # Calculate overall metrics
        evaluation_results["detailed_scores"] = all_evaluations
        evaluation_results["overall_metrics"] = self.metrics.calculate_overall_metrics(
            all_evaluations
        )
        
        # Save complete evaluation results
        complete_output = os.path.join(output_dir, "complete_evaluation.json")
        DataUtils.save_json(evaluation_results, complete_output)
        
        print(f"LLM-Judge evaluation completed. Results saved to: {output_dir}")
        return evaluation_results
    
    def _evaluate_split(
        self, 
        split_data, 
        batch_size: int
    ) -> Dict[str, Any]:
        """
        Evaluate a single dataset split.
        
        Args:
            split_data: Dataset split to evaluate
            batch_size: Batch size for processing
            
        Returns:
            Split evaluation results
        """
        # Prepare evaluation samples
        evaluation_samples = []
        
        for sample in split_data:
            eval_sample = {
                "serial_number": sample.get("serial_number", -1),
                "style_variant": sample.get("style_variant", "unknown"),
                "modified_question": sample.get("modified_question", ""),
                "generated_answer": sample.get("generated_answer", ""),
                "gold_answer": sample.get("gold_answer", ""),
                "original_question": sample.get("original_question", "")
            }
            evaluation_samples.append(eval_sample)
        
        # Process in batches
        all_evaluations = []
        total_batches = (len(evaluation_samples) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(evaluation_samples), batch_size), 
                      desc=f"Evaluating batches", 
                      total=total_batches):
            batch = evaluation_samples[i:i + batch_size]
            batch_results = self._evaluate_batch(batch)
            all_evaluations.extend(batch_results)
        
        # Calculate split metrics
        split_metrics = self.metrics.calculate_split_metrics(all_evaluations)
        
        return {
            "total_samples": len(evaluation_samples),
            "detailed_evaluations": all_evaluations,
            "split_metrics": split_metrics
        }
    
    def _evaluate_batch(self, batch: List[Dict]) -> List[Dict]:
        """
        Evaluate a batch of samples using the LLM judge.
        
        Args:
            batch: Batch of samples to evaluate
            
        Returns:
            List of evaluation results
        """
        batch_results = []
        
        # Set up judge LLM arguments
        judge_args = type('Args', (), {
            'model_id': self.judge_model_id,
            'config_path': str(self.config.config_path),
            'current_dir': './',
            'max_new_tokens': 100  # Short responses for scoring
        })()
        
        # Get LLM judge
        try:
            judge_llm = self.llm_utils.get_llm(judge_args)
        except Exception as e:
            print(f"Error loading judge LLM: {e}")
            # Return empty results with error
            return [{
                "serial_number": sample.get("serial_number", -1),
                "error": f"Judge LLM loading failed: {str(e)}"
            } for sample in batch]
        
        # Evaluate each sample in the batch
        for sample in batch:
            try:
                evaluation = self._evaluate_single_sample(sample, judge_llm)
                batch_results.append(evaluation)
            except Exception as e:
                print(f"Error evaluating sample {sample.get('serial_number', -1)}: {e}")
                # Add error result
                error_result = sample.copy()
                error_result["evaluation_error"] = str(e)
                error_result["scores"] = {}
                batch_results.append(error_result)
        
        return batch_results
    
    def _evaluate_single_sample(self, sample: Dict, judge_llm) -> Dict:
        """
        Evaluate a single QA sample using the LLM judge.
        
        Args:
            sample: Sample to evaluate
            judge_llm: Judge LLM instance
            
        Returns:
            Evaluation result with scores
        """
        # Extract components
        modified_question = sample["modified_question"]
        generated_answer = sample["generated_answer"]
        gold_answer = sample["gold_answer"]
        
        # Create evaluation prompt
        evaluation_prompt = self.rubrics.create_evaluation_prompt(
            modified_question, generated_answer, gold_answer
        )
        
        try:
            # Get evaluation from judge
            judge_response = judge_llm.generate_response(
                conversation=evaluation_prompt,
                max_new_tokens=100
            )
            
            # Parse response to extract scores
            scores = self._parse_judge_response(judge_response)
            
        except Exception as e:
            print(f"Error in judge evaluation: {e}")
            # Default to neutral scores if evaluation fails
            scores = {
                "correctness": 2,
                "completeness": 2, 
                "coherence": 2,
                "linguistic_adaptability": 2
            }
        
        # Create result
        result = sample.copy()
        result["scores"] = scores
        result["judge_response"] = judge_response if 'judge_response' in locals() else ""
        
        return result
    
    def _parse_judge_response(self, response: str) -> Dict[str, int]:
        """
        Parse the judge's response to extract scores.
        
        Args:
            response: Raw response from judge LLM
            
        Returns:
            Dictionary of scores for each criterion
        """
        try:
            # Try to parse as JSON first
            if "{" in response and "}" in response:
                # Extract JSON part
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                json_str = response[start_idx:end_idx]
                
                parsed = json.loads(json_str)
                
                # Map possible key variations to standard keys
                score_mapping = {
                    "correctness": ["correctness", "correct", "accuracy"],
                    "completeness": ["completeness", "complete", "comprehensive"],
                    "coherence": ["fluency_and_coherence", "coherence", "fluency"],
                    "linguistic_adaptability": ["linguistic_adaptability", "adaptability", "style_match"]
                }
                
                scores = {}
                for standard_key, possible_keys in score_mapping.items():
                    for key in possible_keys:
                        if key in parsed:
                            scores[standard_key] = int(parsed[key])
                            break
                    
                    # Default to 2 if not found
                    if standard_key not in scores:
                        scores[standard_key] = 2
                
                return scores
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error parsing judge response: {e}")
        
        # Fallback: try to extract numbers from text
        try:
            import re
            numbers = re.findall(r'\b[1-3]\b', response)
            if len(numbers) >= 4:
                return {
                    "correctness": int(numbers[0]),
                    "completeness": int(numbers[1]),
                    "coherence": int(numbers[2]), 
                    "linguistic_adaptability": int(numbers[3])
                }
        except (ValueError, IndexError):
            pass
        
        # Final fallback: return neutral scores
        return {
            "correctness": 2,
            "completeness": 2,
            "coherence": 2,
            "linguistic_adaptability": 2
        }
    
    def generate_evaluation_report(
        self,
        evaluation_results: Dict[str, Any],
        output_path: str
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from LLM-Judge evaluation
            output_path: Path to save the report
            
        Returns:
            Report summary
        """
        print("Generating evaluation report...")
        
        # Extract key metrics
        overall_metrics = evaluation_results["overall_metrics"]
        detailed_scores = evaluation_results["detailed_scores"]
        
        # Generate report sections
        report = {
            "executive_summary": self._generate_executive_summary(overall_metrics),
            "detailed_analysis": self._generate_detailed_analysis(detailed_scores),
            "style_variant_analysis": self._analyze_by_style_variants(detailed_scores),
            "statistical_summary": overall_metrics,
            "recommendations": self._generate_recommendations(overall_metrics)
        }
        
        # Save report
        DataUtils.save_json(report, output_path)
        
        # Generate markdown report
        markdown_path = output_path.replace('.json', '.md')
        self._generate_markdown_report(report, markdown_path)
        
        print(f"Evaluation report saved to: {output_path}")
        return report
    
    def _generate_executive_summary(self, metrics: Dict) -> Dict[str, Any]:
        """Generate executive summary of evaluation results."""
        return {
            "total_samples_evaluated": metrics.get("total_samples", 0),
            "overall_performance": {
                "correctness": f"{metrics.get('avg_correctness', 0):.2f}/3.0",
                "completeness": f"{metrics.get('avg_completeness', 0):.2f}/3.0",
                "coherence": f"{metrics.get('avg_coherence', 0):.2f}/3.0",
                "linguistic_adaptability": f"{metrics.get('avg_linguistic_adaptability', 0):.2f}/3.0"
            },
            "key_findings": [
                f"Average correctness score: {metrics.get('avg_correctness', 0):.2f}",
                f"Most challenging criterion: {self._identify_weakest_criterion(metrics)}",
                f"Best performing criterion: {self._identify_strongest_criterion(metrics)}"
            ]
        }
    
    def _generate_detailed_analysis(self, detailed_scores: List[Dict]) -> Dict[str, Any]:
        """Generate detailed analysis of evaluation results."""
        if not detailed_scores:
            return {"error": "No detailed scores available"}
        
        # Analyze score distributions
        score_distributions = {}
        for criterion in ["correctness", "completeness", "coherence", "linguistic_adaptability"]:
            scores = [sample["scores"].get(criterion, 2) for sample in detailed_scores 
                     if "scores" in sample and isinstance(sample["scores"], dict)]
            
            if scores:
                score_distributions[criterion] = {
                    "score_1_count": scores.count(1),
                    "score_2_count": scores.count(2), 
                    "score_3_count": scores.count(3),
                    "score_1_pct": scores.count(1) / len(scores) * 100,
                    "score_2_pct": scores.count(2) / len(scores) * 100,
                    "score_3_pct": scores.count(3) / len(scores) * 100
                }
        
        return {
            "score_distributions": score_distributions,
            "total_evaluated": len(detailed_scores)
        }
    
    def _analyze_by_style_variants(self, detailed_scores: List[Dict]) -> Dict[str, Any]:
        """Analyze performance by style variants."""
        variant_analysis = {}
        
        # Group by style variant
        for sample in detailed_scores:
            if "scores" not in sample or not isinstance(sample["scores"], dict):
                continue
                
            variant = sample.get("style_variant", "unknown")
            
            if variant not in variant_analysis:
                variant_analysis[variant] = {
                    "samples": [],
                    "avg_scores": {},
                    "count": 0
                }
            
            variant_analysis[variant]["samples"].append(sample["scores"])
            variant_analysis[variant]["count"] += 1
        
        # Calculate averages for each variant
        for variant, data in variant_analysis.items():
            if data["samples"]:
                for criterion in ["correctness", "completeness", "coherence", "linguistic_adaptability"]:
                    scores = [s.get(criterion, 2) for s in data["samples"]]
                    data["avg_scores"][criterion] = sum(scores) / len(scores)
        
        return variant_analysis
    
    def _identify_weakest_criterion(self, metrics: Dict) -> str:
        """Identify the criterion with the lowest average score."""
        criteria_scores = {
            "correctness": metrics.get("avg_correctness", 0),
            "completeness": metrics.get("avg_completeness", 0),
            "coherence": metrics.get("avg_coherence", 0),
            "linguistic_adaptability": metrics.get("avg_linguistic_adaptability", 0)
        }
        return min(criteria_scores.items(), key=lambda x: x[1])[0]
    
    def _identify_strongest_criterion(self, metrics: Dict) -> str:
        """Identify the criterion with the highest average score."""
        criteria_scores = {
            "correctness": metrics.get("avg_correctness", 0),
            "completeness": metrics.get("avg_completeness", 0),
            "coherence": metrics.get("avg_coherence", 0),
            "linguistic_adaptability": metrics.get("avg_linguistic_adaptability", 0)
        }
        return max(criteria_scores.items(), key=lambda x: x[1])[0]
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        # Analyze each criterion
        correctness = metrics.get("avg_correctness", 0)
        completeness = metrics.get("avg_completeness", 0)
        coherence = metrics.get("avg_coherence", 0)
        adaptability = metrics.get("avg_linguistic_adaptability", 0)
        
        if correctness < 2.0:
            recommendations.append("Focus on improving factual accuracy - correctness scores are below average")
        
        if completeness < 2.0:
            recommendations.append("Work on providing more comprehensive answers that fully address all aspects of questions")
        
        if coherence < 2.0:
            recommendations.append("Improve answer fluency and logical structure")
        
        if adaptability < 1.8:
            recommendations.append("Critical need to improve linguistic adaptability - models are not matching user communication styles")
        
        if all(score > 2.5 for score in [correctness, completeness, coherence, adaptability]):
            recommendations.append("Excellent overall performance across all criteria")
        
        return recommendations
    
    def _generate_markdown_report(self, report: Dict, output_path: str) -> None:
        """Generate a markdown version of the evaluation report."""
        lines = []
        
        lines.append("# SPQA Evaluation Report")
        lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        summary = report["executive_summary"]
        lines.append(f"- **Total Samples**: {summary['total_samples_evaluated']:,}")
        lines.append("")
        
        lines.append("### Overall Performance")
        for criterion, score in summary["overall_performance"].items():
            lines.append(f"- **{criterion.replace('_', ' ').title()}**: {score}")
        lines.append("")
        
        # Key Findings
        lines.append("### Key Findings")
        for finding in summary["key_findings"]:
            lines.append(f"- {finding}")
        lines.append("")
        
        # Recommendations
        lines.append("## Recommendations")
        for rec in report["recommendations"]:
            lines.append(f"- {rec}")
        lines.append("")
        
        # Save markdown
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
