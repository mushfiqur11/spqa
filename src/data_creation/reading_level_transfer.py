"""
Reading level-based style transfer for SPQA framework.

This module handles elementary/middle/high/graduate level style transfers
as part of the SPQA framework's Automated Style Transfer (AST) component.
"""

import os
import re
from typing import Dict, Any, Optional
from tqdm import tqdm
from datasets import DatasetDict

from ..utils.config import Config
from ..utils.llm_utils import LLMUtils
from ..utils.data_utils import DataUtils


class ReadingLevelTransfer:
    """
    Handles reading level-based style transfers.
    
    This class implements the reading level dimension of style transfer, including:
    - Elementary school level
    - Middle school level  
    - High school level
    - Graduate school level
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize ReadingLevelTransfer.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or Config()
        self.llm_utils = LLMUtils(self.config)
        
        # Define supported reading level variants
        self.reading_levels = ["elementary", "middle", "high", "graduate"]
    
    def apply_transfer(
        self, 
        dataset: DatasetDict, 
        args,
        output_dir: str
    ) -> DatasetDict:
        """
        Apply reading level-based style transfers to the entire dataset.
        
        Args:
            dataset: Input dataset
            args: Arguments with model configuration
            output_dir: Output directory for saving results
            
        Returns:
            Dataset with reading level variants added
        """
        print("Applying reading level-based style transfers...")
        
        # Load LLM for style transfer
        llm = self.llm_utils.get_llm(args)
        
        # Apply all reading level variants
        for level in tqdm(self.reading_levels, desc="Reading levels"):
            # Apply transfer to all splits
            dataset = dataset.map(
                lambda sample: self._transfer_sample(
                    sample, level, llm, args
                ),
                desc=f"Applying {level} level transfer"
            )
            
            # Save intermediate results for this level
            level_output = os.path.join(output_dir, f"reading_level_{level}")
            DataUtils.save_dataset(dataset, level_output, format="json")
        
        print("Reading level-based transfers completed.")
        return dataset
    
    def _transfer_sample(
        self, 
        sample: Dict[str, Any], 
        level: str, 
        llm, 
        args
    ) -> Dict[str, Any]:
        """
        Transfer a single sample to the specified reading level.
        
        Args:
            sample: Dataset sample containing the question
            level: Reading level (elementary, middle, high, graduate)
            llm: LLM instance for generation
            args: Arguments with generation parameters
            
        Returns:
            Sample with reading level variant added
        """
        original_question = sample["question"]
        
        # Get system prompt for this level
        system_prompt = self._get_system_prompt(level)
        
        # Format conversation
        conversation = LLMUtils.format_conversation(original_question, system_prompt)
        
        try:
            # Generate transferred question
            transferred_question = llm.generate_response(
                conversation=conversation,
                max_new_tokens=getattr(args, 'max_new_tokens', 200)
            )
            
            # Clean up the response (remove any extra whitespace)
            transferred_question = transferred_question.strip()
            
            # Add to sample
            sample[level] = transferred_question
            
        except Exception as e:
            print(f"Error transferring sample to {level}: {e}")
            # Fallback to original question if transfer fails
            sample[level] = original_question
        
        return sample
    
    def _get_system_prompt(self, level: str) -> str:
        """
        Get the appropriate system prompt for the reading level transfer.
        
        Args:
            level: Reading level (elementary, middle, high, graduate)
            
        Returns:
            System prompt string
        """
        base_prompt = (
            "You are a sentence converter. You will be given a medical question from reddit. "
            "Do not change the meaning of the question or generate anything extra."
        )
        
        level_descriptions = {
            "elementary": (
                f"{base_prompt} Rewrite it as if you were an elementary school student. "
                "The reading-level of the sentence should match elementary level students."
            ),
            "middle": (
                f"{base_prompt} Rewrite it as if you were a middle school student. "
                "The reading-level of the sentence should match middle level students."
            ),
            "high": (
                f"{base_prompt} Rewrite it as if you were a high school student. "
                "The reading-level of the sentence should match high level students."
            ),
            "graduate": (
                f"{base_prompt} Rewrite it as if you were a graduate school student. "
                "The reading-level of the sentence should match graduate level students."
            )
        }
        
        return level_descriptions.get(level, f"{base_prompt} Rewrite it at {level} level.")
    
    def validate_reading_level_transfer(
        self, 
        original: str, 
        transferred: str, 
        level: str
    ) -> Dict[str, Any]:
        """
        Validate the quality of a reading level transfer.
        
        Args:
            original: Original question
            transferred: Transferred question
            level: Reading level applied
            
        Returns:
            Dictionary with validation metrics
        """
        validation_result = {
            "original": original,
            "transferred": transferred,
            "level": level,
            "length_change": len(transferred) - len(original),
            "length_ratio": len(transferred) / len(original) if len(original) > 0 else 0,
            "identical": original.strip() == transferred.strip(),
        }
        
        # Basic quality checks
        validation_result["has_content"] = len(transferred.strip()) > 0
        validation_result["reasonable_length"] = 10 <= len(transferred) <= 1000
        
        # Reading level specific metrics
        validation_result["readability_metrics"] = self._calculate_readability_metrics(transferred)
        validation_result["level_indicators"] = self._check_level_indicators(transferred, level)
        
        return validation_result
    
    def _calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """
        Calculate basic readability metrics for the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with readability metrics
        """
        if not text.strip():
            return {"avg_word_length": 0, "avg_sentence_length": 0, "complex_words": 0}
        
        # Split into sentences and words
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Calculate metrics
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Count complex words (more than 6 characters, simple heuristic)
        complex_words = len([word for word in words if len(word) > 6])
        complex_word_ratio = complex_words / len(words) if words else 0
        
        return {
            "avg_word_length": round(avg_word_length, 2),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "complex_words": complex_words,
            "complex_word_ratio": round(complex_word_ratio, 3),
            "total_words": len(words),
            "total_sentences": len(sentences)
        }
    
    def _check_level_indicators(self, text: str, level: str) -> Dict[str, bool]:
        """
        Check for reading level specific indicators.
        
        Args:
            text: Text to check
            level: Target reading level
            
        Returns:
            Dictionary with level-specific indicators
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        if level == "elementary":
            return self._check_elementary_indicators(text_lower, words)
        elif level == "middle":
            return self._check_middle_indicators(text_lower, words)
        elif level == "high":
            return self._check_high_indicators(text_lower, words)
        elif level == "graduate":
            return self._check_graduate_indicators(text_lower, words)
        else:
            return {}
    
    def _check_elementary_indicators(self, text: str, words: list) -> Dict[str, bool]:
        """Check for elementary level language indicators."""
        simple_words = [
            "hurt", "sick", "help", "doctor", "medicine", "feel", "bad", "good",
            "pain", "head", "stomach", "arm", "leg", "hot", "cold"
        ]
        
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        return {
            "uses_simple_words": any(word in words for word in simple_words),
            "short_avg_word_length": avg_word_length < 5,
            "basic_sentence_structure": not any(word in text for word in ["however", "therefore", "furthermore"]),
            "avoids_complex_medical_terms": not any(
                word in text for word in ["diagnosis", "pathology", "etiology", "prognosis"]
            )
        }
    
    def _check_middle_indicators(self, text: str, words: list) -> Dict[str, bool]:
        """Check for middle school level language indicators."""
        intermediate_words = [
            "symptoms", "treatment", "condition", "problem", "issue", "concerned",
            "appointment", "medication", "infection", "injury"
        ]
        
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        return {
            "uses_intermediate_vocabulary": any(word in words for word in intermediate_words),
            "moderate_word_length": 4 <= avg_word_length <= 6,
            "some_complex_sentences": any(word in text for word in ["because", "although", "since"]),
        }
    
    def _check_high_indicators(self, text: str, words: list) -> Dict[str, bool]:
        """Check for high school level language indicators."""
        advanced_words = [
            "diagnosis", "chronic", "acute", "symptoms", "treatment", "medical",
            "professional", "healthcare", "specialist", "examination"
        ]
        
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        return {
            "uses_advanced_vocabulary": any(word in words for word in advanced_words),
            "longer_word_length": avg_word_length > 5,
            "complex_sentence_structure": any(
                word in text for word in ["however", "therefore", "consequently", "furthermore"]
            ),
        }
    
    def _check_graduate_indicators(self, text: str, words: list) -> Dict[str, bool]:
        """Check for graduate level language indicators."""
        technical_words = [
            "pathology", "etiology", "prognosis", "differential", "clinical",
            "therapeutic", "pharmacological", "physiological", "anatomical",
            "histological", "biochemical", "molecular"
        ]
        
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        complex_words = len([word for word in words if len(word) > 7])
        
        return {
            "uses_technical_vocabulary": any(word in words for word in technical_words),
            "high_avg_word_length": avg_word_length > 6,
            "many_complex_words": complex_words / len(words) > 0.3 if words else False,
            "sophisticated_syntax": any(
                phrase in text for phrase in [
                    "in consideration of", "with regard to", "furthermore", 
                    "notwithstanding", "consequently"
                ]
            ),
        }
