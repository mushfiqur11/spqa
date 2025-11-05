"""
Evaluation rubrics for SPQA framework.

This module defines the evaluation criteria and rubrics used by the LLM-Judge
to assess QA performance across the four dimensions defined in the SPQA paper.
"""

from typing import Dict, List, Any


class EvaluationRubrics:
    """
    Evaluation rubrics and criteria definitions for SPQA framework.
    
    This class encapsulates the structured evaluation criteria based on
    the SPQA paper's methodology for assessing QA performance.
    """
    
    def __init__(self):
        """Initialize evaluation rubrics."""
        self.criteria_definitions = self._get_criteria_definitions()
        self.system_prompt = self._get_system_prompt()
        self.user_prompt_template = self._get_user_prompt_template()
    
    def _get_criteria_definitions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed definitions for each evaluation criterion.
        
        Returns:
            Dictionary with detailed criterion definitions
        """
        return {
            "correctness": {
                "name": "Correctness",
                "description": "Measures the factual correctness and accuracy of the LLM generated response considering the gold answer as factually correct.",
                "scale": {
                    1: "The generated answer is completely incorrect. It does not have any medically accurate information.",
                    2: "The generated answer is mostly incorrect. It has very little medically correct advice or information.",
                    3: "The generated answer is mostly correct."
                },
                "focus": ["Factual alignment", "Medical accuracy", "Information correctness"]
            },
            "completeness": {
                "name": "Completeness", 
                "description": "Evaluates what portion of the question is fully answered by the LLM-generated response.",
                "scale": {
                    1: "The generated answer is absolutely irrelevant and does not answer any of the queries made in the question.",
                    2: "The generated answer is somewhat incomplete, missing the essential details required to address the question.",
                    3: "The generated answer answers the question sufficiently. Important parts of the question are addressed by the generated answer."
                },
                "focus": ["Information coverage", "Question addressing", "Thoroughness"]
            },
            "coherence": {
                "name": "Coherence and Fluency",
                "description": "Assesses the grammatical correctness and logical coherence of the generated response.",
                "scale": {
                    1: "The generated answer is poorly written, with significant grammatical and structural issues.",
                    2: "The generated answer is somewhat fluent but contains noticeable issues and grammatical errors.",
                    3: "The generated answer is fluent and logically structured."
                },
                "focus": ["Grammatical correctness", "Logical structure", "Language fluency"]
            },
            "linguistic_adaptability": {
                "name": "Linguistic Adaptability",
                "description": "Measures how well an LLM adjusts its response based on variations in tone, and user expertise while preserving factuality.",
                "scale": {
                    1: "The generated answer fails to match the tone or style of the modified question and would be totally unreadable for the user asking the question.",
                    2: "The generated answer somewhat matches the tone or style but would not be fully legible for the user asking the question.",
                    3: "The generated answer is appropriate and easy to read for someone who asked the question."
                },
                "focus": ["Tone matching", "Style adaptation", "User appropriateness", "Readability alignment"]
            }
        }
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the LLM judge.
        
        Returns:
            System prompt string
        """
        return """You are a helpful assistant that evaluates medical QA samples.

For each sample, you must evaluate the generated answer (A_gen) as a response to the modified question (Q_mod). Use the gold standard answer (A_gold) as the medically accurate information regarding the topic.

In this QA task, the generated answer (A_gen) and the gold standard answer (A_gold) may come from different linguistic distribution. Do not penalize A_gen for being linguistically different from A_gold.

Use the following four criteria. For each criterion, assign a score from 1 (lowest) to 3 (highest).

1. Correctness: Considering A_gold as medically correct, what portion of A_gen is accurate in answering the Q_mod? A_gen can be significantly different from A_gold.
   - Score 1: A_gen is completely incorrect. It does not have any medically accurate information.
   - Score 2: A_gen is mostly incorrect. It has very little medically correct advice or information.
   - Score 3: The generated answer is mostly correct.

2. Completeness: What portion of the queries made in Q_mod are answered by A_gen?
   - Score 1: A_gen is absolutely irrelevant and does not answer any of the queries made in Q_mod.
   - Score 2: A_gen is somewhat incomplete, missing the essential details required to address Q_mod.
   - Score 3: A_gen answers Q_mod sufficiently. Important parts of the question in Q_mod are addressed by A_gen.

3. Fluency and Coherence: How well is A_gen written in terms of language fluency and logical structure?
   - Score 1: A_gen is poorly written, with significant grammatical and structural issues.
   - Score 2: A_gen is somewhat fluent but contains noticeable issues and grammatical errors.
   - Score 3: A_gen is fluent and logically structured.

4. Linguistic Adaptability: How effectively does A_gen adopt the style and tone of the modified question (Q_mod)?
   - Score 1: A_gen fails to match the tone or style of Q_mod and would be totally unreadable for the user asking Q_mod.
   - Score 2: A_gen somewhat matches the tone or style but would not be fully legible for the user asking Q_mod.
   - Score 3: A_gen is appropriate and easy to read for someone who asked the question Q_mod.

Return your evaluation in JSON format as follows:
{
    "correctness": <rating as an integer>,
    "completeness": <rating as an integer>,
    "coherence": <rating as an integer>,
    "linguistic_adaptability": <rating as an integer>
}

Ensure that your output contains only the JSON object."""
    
    def _get_user_prompt_template(self) -> str:
        """
        Get the user prompt template for evaluation.
        
        Returns:
            User prompt template
        """
        return """Evaluate the following QA sample:

Modified Question (Q_mod): [SEP] {question} [SEP]
Generated Answer (A_gen): [SEP] {answer} [SEP]
Gold Standard Answer (A_gold): [SEP] {gold} [SEP]"""
    
    def create_evaluation_prompt(
        self, 
        question: str, 
        answer: str, 
        gold_answer: str
    ) -> List[Dict[str, str]]:
        """
        Create a complete evaluation prompt for the LLM judge.
        
        Args:
            question: The modified/style-transferred question
            answer: The generated answer to evaluate
            gold_answer: The gold standard answer
            
        Returns:
            Conversation formatted for LLM input
        """
        # Format the user prompt
        user_content = self.user_prompt_template.format(
            question=question.strip(),
            answer=answer.strip(),
            gold=gold_answer.strip()
        )
        
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
    
    def get_criterion_definition(self, criterion: str) -> Dict[str, Any]:
        """
        Get detailed definition for a specific criterion.
        
        Args:
            criterion: Name of the criterion
            
        Returns:
            Criterion definition dictionary
        """
        return self.criteria_definitions.get(criterion, {})
    
    def get_all_criteria(self) -> List[str]:
        """
        Get list of all evaluation criteria.
        
        Returns:
            List of criterion names
        """
        return list(self.criteria_definitions.keys())
    
    def get_criterion_scale(self, criterion: str) -> Dict[int, str]:
        """
        Get the scoring scale for a specific criterion.
        
        Args:
            criterion: Name of the criterion
            
        Returns:
            Dictionary mapping scores to descriptions
        """
        return self.criteria_definitions.get(criterion, {}).get("scale", {})
    
    def validate_scores(self, scores: Dict[str, int]) -> Dict[str, Any]:
        """
        Validate evaluation scores against the rubric.
        
        Args:
            scores: Dictionary of scores to validate
            
        Returns:
            Validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        required_criteria = set(self.criteria_definitions.keys())
        provided_criteria = set(scores.keys())
        
        # Check for missing criteria
        missing = required_criteria - provided_criteria
        if missing:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Missing criteria: {missing}")
        
        # Check for extra criteria
        extra = provided_criteria - required_criteria
        if extra:
            validation_result["warnings"].append(f"Extra criteria provided: {extra}")
        
        # Check score ranges
        for criterion, score in scores.items():
            if criterion in self.criteria_definitions:
                if not isinstance(score, int) or score < 1 or score > 3:
                    validation_result["valid"] = False
                    validation_result["errors"].append(
                        f"Invalid score for {criterion}: {score} (must be 1, 2, or 3)"
                    )
        
        return validation_result
    
    def normalize_scores(self, scores: Dict[str, int]) -> Dict[str, float]:
        """
        Normalize scores to 0-1 range for analysis.
        
        Args:
            scores: Raw scores (1-3 scale)
            
        Returns:
            Normalized scores (0-1 scale)
        """
        return {
            criterion: (score - 1) / 2.0  # Convert 1-3 to 0-1
            for criterion, score in scores.items()
        }
    
    def get_evaluation_summary_template(self) -> str:
        """
        Get template for evaluation summary reports.
        
        Returns:
            Report template string
        """
        return """
# Evaluation Summary

## Overall Performance
- **Correctness**: {avg_correctness:.2f}/3.0 ({norm_correctness:.1%})
- **Completeness**: {avg_completeness:.2f}/3.0 ({norm_completeness:.1%})
- **Coherence and Fluency**: {avg_coherence:.2f}/3.0 ({norm_coherence:.1%})
- **Linguistic Adaptability**: {avg_linguistic_adaptability:.2f}/3.0 ({norm_linguistic_adaptability:.1%})

## Score Distributions
{score_distributions}

## Key Insights
{key_insights}

## Recommendations
{recommendations}
"""
    
    def create_human_readable_rubric(self) -> str:
        """
        Create a human-readable version of the evaluation rubric.
        
        Returns:
            Formatted rubric string
        """
        lines = ["# SPQA Evaluation Rubric", ""]
        
        for criterion_name, criterion_data in self.criteria_definitions.items():
            lines.append(f"## {criterion_data['name']}")
            lines.append(f"**Description**: {criterion_data['description']}")
            lines.append("")
            lines.append("**Scoring Scale**:")
            
            for score, description in criterion_data["scale"].items():
                lines.append(f"- **Score {score}**: {description}")
            lines.append("")
            
            lines.append(f"**Focus Areas**: {', '.join(criterion_data['focus'])}")
            lines.append("")
        
        return "\n".join(lines)
    
    def export_rubric_json(self) -> Dict[str, Any]:
        """
        Export the complete rubric as a JSON-serializable dictionary.
        
        Returns:
            Complete rubric data
        """
        return {
            "rubric_version": "1.0",
            "criteria": self.criteria_definitions,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "evaluation_instructions": {
                "score_range": "1-3 (1=lowest, 3=highest)",
                "output_format": "JSON with criterion scores",
                "special_instructions": [
                    "Consider gold answer as medically accurate reference",
                    "Do not penalize linguistic differences between generated and gold answers",
                    "Focus on appropriateness for the style-transferred question context"
                ]
            }
        }
