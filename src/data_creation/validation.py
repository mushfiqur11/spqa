"""
Style transfer validation for SPQA framework.

This module provides validation utilities for assessing the quality of style transfers,
ensuring that semantic meaning is preserved while stylistic changes are applied.
"""

import statistics
from typing import Dict, List, Any, Optional, Tuple
from datasets import DatasetDict

from ..utils.config import Config
from ..utils.data_utils import DataUtils


class StyleTransferValidator:
    """
    Validator for style transfer quality assessment.
    
    This class implements validation methods to ensure that:
    1. Style transfer was successful (achieved target style)
    2. Semantic meaning was preserved during transfer
    3. Generated text quality is acceptable
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize StyleTransferValidator.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or Config()
    
    def validate_dataset(
        self, 
        dataset: DatasetDict, 
        style_variants: List[str]
    ) -> Dict[str, Any]:
        """
        Validate style transfers across an entire dataset.
        
        Args:
            dataset: Dataset with style variants
            style_variants: List of style variants to validate
            
        Returns:
            Comprehensive validation results
        """
        print("Starting dataset-wide style transfer validation...")
        
        validation_results = {
            "overall_stats": {},
            "variant_stats": {},
            "sample_validations": {}
        }
        
        total_samples = 0
        all_success_rates = []
        all_preservation_rates = []
        
        # Validate each split
        for split_name, split_data in dataset.items():
            print(f"Validating split: {split_name}")
            
            split_results = self._validate_split(split_data, style_variants)
            validation_results["sample_validations"][split_name] = split_results
            
            # Aggregate statistics
            total_samples += len(split_data)
            all_success_rates.extend(split_results["success_rates"])
            all_preservation_rates.extend(split_results["preservation_rates"])
        
        # Calculate overall statistics
        validation_results["overall_stats"] = {
            "total_samples": total_samples,
            "total_variants": len(style_variants),
            "avg_style_success_rate": statistics.mean(all_success_rates) if all_success_rates else 0,
            "avg_meaning_preservation_rate": statistics.mean(all_preservation_rates) if all_preservation_rates else 0,
            "style_success_std": statistics.stdev(all_success_rates) if len(all_success_rates) > 1 else 0,
            "meaning_preservation_std": statistics.stdev(all_preservation_rates) if len(all_preservation_rates) > 1 else 0,
        }
        
        # Calculate per-variant statistics
        validation_results["variant_stats"] = self._calculate_variant_stats(
            dataset, style_variants
        )
        
        print("Dataset validation completed.")
        return validation_results
    
    def _validate_split(
        self, 
        split_data, 
        style_variants: List[str]
    ) -> Dict[str, Any]:
        """
        Validate style transfers for a single dataset split.
        
        Args:
            split_data: Dataset split to validate
            style_variants: Style variants to validate
            
        Returns:
            Validation results for the split
        """
        sample_results = []
        success_rates = []
        preservation_rates = []
        
        for sample in split_data:
            original_question = sample.get("question", "")
            
            sample_validation = {
                "serial_number": sample.get("serial_number", -1),
                "original_question": original_question,
                "variants": {}
            }
            
            variant_successes = []
            variant_preservations = []
            
            # Validate each style variant
            for variant in style_variants:
                if variant in sample:
                    transferred_question = sample[variant]
                    
                    # Validate this specific transfer
                    variant_result = self.validate_single_transfer(
                        original_question, transferred_question, variant
                    )
                    
                    sample_validation["variants"][variant] = variant_result
                    
                    # Track success metrics
                    variant_successes.append(variant_result["style_success_score"])
                    variant_preservations.append(variant_result["meaning_preservation_score"])
            
            # Calculate sample-level statistics
            if variant_successes:
                sample_validation["avg_style_success"] = statistics.mean(variant_successes)
                sample_validation["avg_meaning_preservation"] = statistics.mean(variant_preservations)
            else:
                sample_validation["avg_style_success"] = 0
                sample_validation["avg_meaning_preservation"] = 0
            
            sample_results.append(sample_validation)
            success_rates.append(sample_validation["avg_style_success"])
            preservation_rates.append(sample_validation["avg_meaning_preservation"])
        
        return {
            "sample_results": sample_results,
            "success_rates": success_rates,
            "preservation_rates": preservation_rates,
            "avg_success_rate": statistics.mean(success_rates) if success_rates else 0,
            "avg_preservation_rate": statistics.mean(preservation_rates) if preservation_rates else 0,
        }
    
    def validate_single_transfer(
        self, 
        original: str, 
        transferred: str, 
        variant: str
    ) -> Dict[str, Any]:
        """
        Validate a single style transfer instance.
        
        Args:
            original: Original question text
            transferred: Style-transferred question text
            variant: Style variant applied
            
        Returns:
            Detailed validation metrics for this transfer
        """
        validation_result = {
            "original": original,
            "transferred": transferred,
            "variant": variant,
            "basic_quality": self._check_basic_quality(transferred),
            "style_indicators": self._check_style_indicators(transferred, variant),
            "meaning_preservation": self._check_meaning_preservation(original, transferred),
            "length_analysis": self._analyze_length_changes(original, transferred),
        }
        
        # Calculate composite scores
        validation_result["style_success_score"] = self._calculate_style_success_score(
            validation_result["style_indicators"]
        )
        validation_result["meaning_preservation_score"] = self._calculate_preservation_score(
            validation_result["meaning_preservation"]
        )
        validation_result["overall_quality_score"] = (
            validation_result["style_success_score"] * 0.4 +
            validation_result["meaning_preservation_score"] * 0.5 +
            validation_result["basic_quality"]["quality_score"] * 0.1
        )
        
        return validation_result
    
    def _check_basic_quality(self, text: str) -> Dict[str, Any]:
        """
        Check basic quality metrics of the transferred text.
        
        Args:
            text: Text to check
            
        Returns:
            Basic quality metrics
        """
        if not text or not text.strip():
            return {
                "has_content": False,
                "reasonable_length": False,
                "ends_properly": False,
                "quality_score": 0.0
            }
        
        text = text.strip()
        
        # Basic checks
        has_content = len(text) > 0
        reasonable_length = 10 <= len(text) <= 1000
        ends_properly = text.endswith(('.', '?', '!')) or text.endswith('...')
        
        # Calculate quality score
        quality_indicators = [has_content, reasonable_length, ends_properly]
        quality_score = sum(quality_indicators) / len(quality_indicators)
        
        return {
            "has_content": has_content,
            "reasonable_length": reasonable_length,
            "ends_properly": ends_properly,
            "character_count": len(text),
            "word_count": len(text.split()),
            "quality_score": quality_score
        }
    
    def _check_style_indicators(self, text: str, variant: str) -> Dict[str, Any]:
        """
        Check for style-specific indicators in the transferred text.
        
        Args:
            text: Transferred text
            variant: Style variant that was applied
            
        Returns:
            Style-specific indicators
        """
        text_lower = text.lower()
        
        if variant == "formal":
            return self._check_formal_style(text, text_lower)
        elif variant == "informal":
            return self._check_informal_style(text, text_lower)
        elif variant in ["elementary", "middle", "high", "graduate"]:
            return self._check_reading_level_style(text, text_lower, variant)
        elif variant == "informed":
            return self._check_informed_style(text, text_lower)
        elif variant == "uninformed":
            return self._check_uninformed_style(text, text_lower)
        else:
            return {"unknown_variant": True}
    
    def _check_formal_style(self, text: str, text_lower: str) -> Dict[str, Any]:
        """Check for formal style indicators."""
        contractions = ["don't", "can't", "won't", "i'm", "it's", "you're"]
        formal_words = ["please", "would", "could", "kindly", "regarding", "concerning"]
        
        return {
            "avoids_contractions": not any(cont in text_lower for cont in contractions),
            "uses_formal_language": any(word in text_lower for word in formal_words),
            "proper_punctuation": text.strip().endswith(('.', '?', '!')),
            "appropriate_capitalization": text[0].isupper() if text else False,
        }
    
    def _check_informal_style(self, text: str, text_lower: str) -> Dict[str, Any]:
        """Check for informal style indicators."""
        contractions = ["don't", "can't", "won't", "i'm", "it's", "you're"]
        casual_words = ["hey", "like", "kinda", "sorta", "pretty", "really"]
        
        return {
            "uses_contractions": any(cont in text_lower for cont in contractions),
            "uses_casual_language": any(word in text_lower for word in casual_words),
            "conversational_tone": "?" in text or "!" in text,
        }
    
    def _check_reading_level_style(self, text: str, text_lower: str, level: str) -> Dict[str, Any]:
        """Check for reading level specific indicators."""
        words = text_lower.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        if level == "elementary":
            simple_words = ["hurt", "sick", "help", "doctor", "feel", "bad", "good"]
            return {
                "uses_simple_vocabulary": any(word in words for word in simple_words),
                "short_words": avg_word_length < 5,
                "avoids_complex_terms": not any(len(word) > 8 for word in words[:5]),
            }
        elif level == "graduate":
            complex_words = ["diagnosis", "pathology", "etiology", "therapeutic"]
            return {
                "uses_technical_vocabulary": any(word in words for word in complex_words),
                "long_words": avg_word_length > 6,
                "sophisticated_language": any(len(word) > 10 for word in words),
            }
        else:
            return {
                "appropriate_complexity": 4 <= avg_word_length <= 7,
                "balanced_vocabulary": True,
            }
    
    def _check_informed_style(self, text: str, text_lower: str) -> Dict[str, Any]:
        """Check for medically informed style indicators."""
        medical_terms = [
            "symptoms", "diagnosis", "treatment", "condition", "medication",
            "chronic", "acute", "syndrome", "pathology"
        ]
        
        return {
            "uses_medical_terminology": any(term in text_lower for term in medical_terms),
            "technical_precision": len([word for word in text_lower.split() if len(word) > 7]) > 0,
        }
    
    def _check_uninformed_style(self, text: str, text_lower: str) -> Dict[str, Any]:
        """Check for medically uninformed style indicators."""
        simple_terms = ["hurt", "pain", "sick", "feel", "bad", "help", "doctor"]
        medical_terms = ["diagnosis", "pathology", "etiology", "syndrome"]
        
        return {
            "uses_simple_language": any(term in text_lower for term in simple_terms),
            "avoids_medical_jargon": not any(term in text_lower for term in medical_terms),
        }
    
    def _check_meaning_preservation(self, original: str, transferred: str) -> Dict[str, Any]:
        """
        Check if semantic meaning was preserved during transfer.
        
        Args:
            original: Original text
            transferred: Transferred text
            
        Returns:
            Meaning preservation metrics
        """
        if not original.strip() or not transferred.strip():
            return {
                "both_have_content": False,
                "similar_length": False,
                "key_concepts_preserved": False,
            }
        
        # Basic similarity checks
        orig_words = set(original.lower().split())
        trans_words = set(transferred.lower().split())
        
        # Calculate word overlap
        common_words = orig_words.intersection(trans_words)
        word_overlap_ratio = len(common_words) / len(orig_words.union(trans_words)) if orig_words.union(trans_words) else 0
        
        # Length similarity
        length_ratio = len(transferred) / len(original) if len(original) > 0 else 0
        similar_length = 0.3 <= length_ratio <= 3.0
        
        # Key medical concepts (simple heuristic)
        medical_keywords = ["pain", "hurt", "sick", "doctor", "medicine", "treatment", "symptoms"]
        orig_medical = [word for word in orig_words if word in medical_keywords]
        trans_medical = [word for word in trans_words if word in medical_keywords]
        medical_preservation = len(set(orig_medical).intersection(set(trans_medical))) / len(orig_medical) if orig_medical else 1.0
        
        return {
            "both_have_content": len(original.strip()) > 0 and len(transferred.strip()) > 0,
            "word_overlap_ratio": round(word_overlap_ratio, 3),
            "similar_length": similar_length,
            "length_ratio": round(length_ratio, 2),
            "medical_concept_preservation": round(medical_preservation, 3),
            "key_concepts_preserved": word_overlap_ratio > 0.2 and medical_preservation > 0.5,
        }
    
    def _analyze_length_changes(self, original: str, transferred: str) -> Dict[str, Any]:
        """
        Analyze length changes between original and transferred text.
        
        Args:
            original: Original text
            transferred: Transferred text
            
        Returns:
            Length analysis metrics
        """
        orig_chars = len(original)
        trans_chars = len(transferred)
        orig_words = len(original.split())
        trans_words = len(transferred.split())
        
        return {
            "original_chars": orig_chars,
            "transferred_chars": trans_chars,
            "char_change": trans_chars - orig_chars,
            "char_ratio": trans_chars / orig_chars if orig_chars > 0 else 0,
            "original_words": orig_words,
            "transferred_words": trans_words,
            "word_change": trans_words - orig_words,
            "word_ratio": trans_words / orig_words if orig_words > 0 else 0,
        }
    
    def _calculate_style_success_score(self, style_indicators: Dict[str, Any]) -> float:
        """Calculate a composite style success score."""
        if "unknown_variant" in style_indicators:
            return 0.0
        
        # Count positive style indicators
        positive_indicators = [v for v in style_indicators.values() if isinstance(v, bool) and v]
        total_indicators = [v for v in style_indicators.values() if isinstance(v, bool)]
        
        if not total_indicators:
            return 0.5  # Neutral score if no boolean indicators
        
        return len(positive_indicators) / len(total_indicators)
    
    def _calculate_preservation_score(self, preservation_metrics: Dict[str, Any]) -> float:
        """Calculate a composite meaning preservation score."""
        weights = {
            "both_have_content": 0.3,
            "key_concepts_preserved": 0.4,
            "similar_length": 0.2,
            "word_overlap_ratio": 0.1
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in preservation_metrics:
                value = preservation_metrics[metric]
                if isinstance(value, bool):
                    score += weight * (1.0 if value else 0.0)
                elif isinstance(value, (int, float)):
                    score += weight * min(1.0, value)  # Cap at 1.0
        
        return score
    
    def _calculate_variant_stats(
        self, 
        dataset: DatasetDict, 
        style_variants: List[str]
    ) -> Dict[str, Any]:
        """Calculate statistics per style variant."""
        variant_stats = {}
        
        for variant in style_variants:
            variant_samples = []
            
            # Collect all samples for this variant across splits
            for split_name, split_data in dataset.items():
                for sample in split_data:
                    if variant in sample and sample[variant]:
                        original = sample.get("question", "")
                        transferred = sample[variant]
                        
                        validation = self.validate_single_transfer(original, transferred, variant)
                        variant_samples.append(validation)
            
            if variant_samples:
                # Calculate statistics for this variant
                style_scores = [s["style_success_score"] for s in variant_samples]
                preservation_scores = [s["meaning_preservation_score"] for s in variant_samples]
                quality_scores = [s["overall_quality_score"] for s in variant_samples]
                
                variant_stats[variant] = {
                    "total_samples": len(variant_samples),
                    "avg_style_success": statistics.mean(style_scores),
                    "avg_meaning_preservation": statistics.mean(preservation_scores),
                    "avg_overall_quality": statistics.mean(quality_scores),
                    "style_success_std": statistics.stdev(style_scores) if len(style_scores) > 1 else 0,
                    "preservation_std": statistics.stdev(preservation_scores) if len(preservation_scores) > 1 else 0,
                }
        
        return variant_stats
