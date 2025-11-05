"""
Formality-based style transfer for SPQA framework.

This module handles formal/informal and domain expert/layperson style transfers
as part of the SPQA framework's Automated Style Transfer (AST) component.
"""

import os
from typing import Dict, Any, Optional
from tqdm import tqdm
from datasets import DatasetDict

from ..utils.config import Config
from ..utils.llm_utils import LLMUtils
from ..utils.data_utils import DataUtils


class FormalityTransfer:
    """
    Handles formality spectrum and domain knowledge style transfers.
    
    This class implements the formality dimension of style transfer, including:
    - Formal vs. Informal style variants
    - Domain expert vs. Layperson (informed vs. uninformed) variants
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize FormalityTransfer.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or Config()
        self.llm_utils = LLMUtils(self.config)
        
        # Define supported formality variants
        self.formality_variants = ["formal", "informal"]
        self.domain_knowledge_variants = ["informed", "uninformed"]
    
    def apply_transfer(
        self, 
        dataset: DatasetDict, 
        args,
        output_dir: str
    ) -> DatasetDict:
        """
        Apply formality-based style transfers to the entire dataset.
        
        Args:
            dataset: Input dataset
            args: Arguments with model configuration
            output_dir: Output directory for saving results
            
        Returns:
            Dataset with formality variants added
        """
        print("Applying formality-based style transfers...")
        
        # Load LLM for style transfer
        llm = self.llm_utils.get_llm(args)
        
        # Apply all formality variants
        all_variants = self.formality_variants + self.domain_knowledge_variants
        
        for variant in tqdm(all_variants, desc="Formality variants"):
            # Determine style type
            if variant in self.formality_variants:
                style_type = "formality"
            else:
                style_type = "domain_knowledge"
            
            # Apply transfer to all splits
            dataset = dataset.map(
                lambda sample: self._transfer_sample(
                    sample, style_type, variant, llm, args
                ),
                desc=f"Applying {variant} transfer"
            )
            
            # Save intermediate results for this variant
            variant_output = os.path.join(output_dir, f"formality_{variant}")
            DataUtils.save_dataset(dataset, variant_output, format="json")
        
        print("Formality-based transfers completed.")
        return dataset
    
    def _transfer_sample(
        self, 
        sample: Dict[str, Any], 
        style_type: str,
        variant: str, 
        llm, 
        args
    ) -> Dict[str, Any]:
        """
        Transfer a single sample to the specified formality variant.
        
        Args:
            sample: Dataset sample containing the question
            style_type: Type of style (formality or domain_knowledge)
            variant: Specific variant (formal, informal, informed, uninformed)
            llm: LLM instance for generation
            args: Arguments with generation parameters
            
        Returns:
            Sample with formality variant added
        """
        original_question = sample["question"]
        
        # Get system prompt for this variant
        system_prompt = self._get_system_prompt(style_type, variant)
        
        # Format conversation
        conversation = LLMUtils.format_conversation(original_question, system_prompt)
        
        try:
            # Generate transferred question
            transferred_question = llm.generate_response(
                conversation=conversation,
                max_new_tokens=getattr(args, 'max_new_tokens', 250)
            )
            
            # Clean up the response (remove any extra whitespace)
            transferred_question = transferred_question.strip()
            
            # Add to sample
            sample[variant] = transferred_question
            
        except Exception as e:
            print(f"Error transferring sample to {variant}: {e}")
            # Fallback to original question if transfer fails
            sample[variant] = original_question
        
        return sample
    
    def _get_system_prompt(self, style_type: str, variant: str) -> str:
        """
        Get the appropriate system prompt for the style transfer.
        
        Args:
            style_type: Type of style (formality or domain_knowledge)
            variant: Specific variant within the style type
            
        Returns:
            System prompt string
        """
        base_prompt = (
            "You are a sentence converter. You will be given a medical question from reddit. "
            "Do not change the meaning of the question or generate anything extra."
        )
        
        if style_type == "formality":
            if variant == "formal":
                return (
                    f"{base_prompt} Rewrite it in a formal tone with serious, "
                    "respectful, and direct writing."
                )
            elif variant == "informal":
                return (
                    f"{base_prompt} Rewrite it in an informal tone with a "
                    "casual way of writing."
                )
        
        elif style_type == "domain_knowledge":
            if variant == "informed":
                return (
                    f"{base_prompt} Rewrite it as if you are medically informed. "
                    "You can use advanced medical terms."
                )
            elif variant == "uninformed":
                return (
                    f"{base_prompt} Rewrite it as if you are a person who is not "
                    "medically well-versed. You cannot use advanced medical terms."
                )
        
        # Fallback
        return f"{base_prompt} Rewrite it in {variant} style."
    
    def validate_formality_transfer(
        self, 
        original: str, 
        transferred: str, 
        variant: str
    ) -> Dict[str, Any]:
        """
        Validate the quality of a formality transfer.
        
        Args:
            original: Original question
            transferred: Transferred question
            variant: Style variant applied
            
        Returns:
            Dictionary with validation metrics
        """
        validation_result = {
            "original": original,
            "transferred": transferred,
            "variant": variant,
            "length_change": len(transferred) - len(original),
            "length_ratio": len(transferred) / len(original) if len(original) > 0 else 0,
            "identical": original.strip() == transferred.strip(),
        }
        
        # Basic quality checks
        validation_result["has_content"] = len(transferred.strip()) > 0
        validation_result["reasonable_length"] = 10 <= len(transferred) <= 1000
        
        # Style-specific validation
        if variant == "formal":
            validation_result["style_indicators"] = self._check_formal_indicators(transferred)
        elif variant == "informal":
            validation_result["style_indicators"] = self._check_informal_indicators(transferred)
        elif variant == "informed":
            validation_result["style_indicators"] = self._check_informed_indicators(transferred)
        elif variant == "uninformed":
            validation_result["style_indicators"] = self._check_uninformed_indicators(transferred)
        
        return validation_result
    
    def _check_formal_indicators(self, text: str) -> Dict[str, bool]:
        """Check for formal language indicators."""
        text_lower = text.lower()
        return {
            "no_contractions": "don't" not in text_lower and "can't" not in text_lower,
            "has_complete_sentences": text.strip().endswith(('.', '?', '!')),
            "proper_capitalization": text[0].isupper() if text else False,
        }
    
    def _check_informal_indicators(self, text: str) -> Dict[str, bool]:
        """Check for informal language indicators."""
        text_lower = text.lower()
        return {
            "has_contractions": any(word in text_lower for word in ["don't", "can't", "won't", "i'm"]),
            "casual_tone": any(word in text_lower for word in ["hey", "like", "kinda", "sorta"]),
        }
    
    def _check_informed_indicators(self, text: str) -> Dict[str, bool]:
        """Check for medically informed language indicators."""
        text_lower = text.lower()
        medical_terms = [
            "diagnosis", "symptoms", "treatment", "medication", "chronic", 
            "acute", "syndrome", "pathology", "etiology", "prognosis"
        ]
        return {
            "has_medical_terms": any(term in text_lower for term in medical_terms),
            "technical_language": len([word for word in text.split() if len(word) > 8]) > 0,
        }
    
    def _check_uninformed_indicators(self, text: str) -> Dict[str, bool]:
        """Check for medically uninformed language indicators."""
        text_lower = text.lower()
        simple_terms = ["hurt", "pain", "sick", "feel", "bad", "good", "help"]
        medical_terms = [
            "diagnosis", "symptoms", "treatment", "chronic", "acute", 
            "syndrome", "pathology", "etiology"
        ]
        return {
            "has_simple_terms": any(term in text_lower for term in simple_terms),
            "avoids_medical_terms": not any(term in text_lower for term in medical_terms),
        }
