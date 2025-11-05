"""
LLM utilities for SPQA framework.

This module provides utilities for working with Large Language Models,
including interfaces to the llm_module dependency and common LLM operations.
"""

import sys
import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

try:
    # Import from llm_module dependency
    from llm_modules.llm_run import llm_ready, model_lookup_table
    LLM_MODULE_AVAILABLE = True
except ImportError:
    LLM_MODULE_AVAILABLE = False
    print("Warning: llm_module not found. Please install it from ../llm_module/")

from .config import Config


class LLMUtils:
    """Utility class for LLM operations in SPQA framework."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize LLM utilities.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        if not LLM_MODULE_AVAILABLE:
            raise ImportError(
                "llm_module is required but not available. "
                "Please install it from ../llm_module/"
            )
        
        self.config = config or Config()
        self._llm_cache = {}
    
    def get_llm(self, args) -> Any:
        """
        Get an LLM instance using llm_module.
        
        Args:
            args: Arguments object with model configuration
            
        Returns:
            LLM instance ready for inference
        """
        # Update args with config if needed
        self.config.update_args(args)
        
        # Create cache key based on model and settings
        cache_key = f"{args.model_id}_{getattr(args, 'quantization', 'none')}"
        
        # Return cached instance if available
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]
        
        # Create new LLM instance
        llm = llm_ready(args)
        self._llm_cache[cache_key] = llm
        
        return llm
    
    def is_model_supported(self, model_id: str) -> bool:
        """
        Check if a model is supported by llm_module.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if model is supported
        """
        if not LLM_MODULE_AVAILABLE:
            return False
        
        return model_id in model_lookup_table
    
    def get_supported_models(self) -> List[str]:
        """
        Get list of all supported models.
        
        Returns:
            List of supported model identifiers
        """
        if not LLM_MODULE_AVAILABLE:
            return []
        
        return list(model_lookup_table.keys())
    
    def get_model_type(self, model_id: str) -> str:
        """
        Get the type of a model (hf-llm or openai-llm).
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model type string
        """
        if not LLM_MODULE_AVAILABLE:
            return "unknown"
        
        return model_lookup_table.get(model_id, "unknown")
    
    @staticmethod
    def format_conversation(question: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Format a question into conversation format for LLM input.
        
        Args:
            question: The question text
            system_prompt: Optional system prompt
            
        Returns:
            Conversation in expected format
        """
        conversation = []
        
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        
        conversation.append({"role": "user", "content": question})
        
        return conversation
    
    @staticmethod
    def get_qa_system_prompt() -> str:
        """
        Get the default system prompt for QA tasks.
        
        Returns:
            System prompt for medical QA
        """
        return (
            "You are a medical expert. Please answer the following question with "
            "medically accurate advice. Keep the answer short and precise. "
            "Try to maintain the same linguistic tone as the question."
        )
    
    @staticmethod
    def get_style_transfer_prompt(style_type: str, style_variant: str) -> str:
        """
        Get system prompt for style transfer tasks.
        
        Args:
            style_type: Type of style transfer (reading_level, formality, domain_knowledge)
            style_variant: Specific variant within the style type
            
        Returns:
            System prompt for style transfer
        """
        base_prompt = (
            "You are a sentence converter. You will be given a medical question from reddit. "
            "Do not change the meaning of the question or generate anything extra."
        )
        
        if style_type == "reading_level":
            if style_variant in ["elementary", "middle", "high", "graduate"]:
                return f"{base_prompt} Rewrite it as if you were an {style_variant} school student. The reading-level of the sentence should match {style_variant} level students."
        
        elif style_type == "formality":
            if style_variant == "formal":
                return f"{base_prompt} Rewrite it in a formal tone with serious, respectful, and direct writing."
            elif style_variant == "informal":
                return f"{base_prompt} Rewrite it in an informal tone with a casual way of writing."
        
        elif style_type == "domain_knowledge":
            if style_variant == "informed":
                return f"{base_prompt} Rewrite it as if you are medically informed. You can use advanced medical terms."
            elif style_variant == "uninformed":
                return f"{base_prompt} Rewrite it as if you are a person who is not medically well-versed. You cannot use advanced medical terms."
        
        # Default fallback
        return f"{base_prompt} Rewrite it in {style_variant} style."
    
    def clear_cache(self) -> None:
        """Clear the LLM instance cache."""
        self._llm_cache.clear()
    
    def __del__(self):
        """Clean up cached LLM instances."""
        self.clear_cache()
