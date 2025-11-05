"""
Configuration management for SPQA framework.

This module handles loading and managing configuration settings for the SPQA framework,
including model paths, API keys, and framework parameters.
"""

import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path


class Config:
    """Configuration manager for SPQA framework."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, loads default config.
        """
        if config_path is None:
            # Default to config.json in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.json"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self._config.get(key, default)
    
    def get_token(self, token_path_key: str) -> str:
        """
        Load token from file path specified in config.
        
        Args:
            token_path_key: Key in config that contains path to token file
            
        Returns:
            Token string
            
        Raises:
            FileNotFoundError: If token file doesn't exist
            KeyError: If token_path_key not in config
        """
        token_path = self._config.get(token_path_key)
        if not token_path:
            raise KeyError(f"Token path key '{token_path_key}' not found in config")
        
        if not os.path.exists(token_path):
            raise FileNotFoundError(f"Token file not found: {token_path}")
        
        with open(token_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    @property
    def hf_token_path(self) -> str:
        """Get Hugging Face token path."""
        return self.get("HF_TOKEN_PATH", "")
    
    @property
    def openai_token_path(self) -> str:
        """Get OpenAI API key path."""
        return self.get("OPENAI_APIKEY_PATH", "")
    
    @property
    def wandb_token_path(self) -> str:
        """Get Weights & Biases token path."""
        return self.get("WANDB_TOKEN_PATH", "")
    
    @property 
    def cache_path(self) -> str:
        """Get cache directory path."""
        return self.get("CACHE_PATH", "./.cache")
    
    @property
    def hf_model_path(self) -> str:
        """Get Hugging Face model cache path."""
        return self.get("HF_MODEL_PATH", "./hf_models")
    
    @property
    def style_variants(self) -> Dict[str, List[str]]:
        """Get supported style variants."""
        return self.get("STYLE_VARIANTS", {
            "reading_level": ["elementary", "middle", "high", "graduate"],
            "formality": ["formal", "informal"], 
            "domain_knowledge": ["informed", "uninformed"]
        })
    
    @property
    def evaluation_criteria(self) -> List[str]:
        """Get evaluation criteria."""
        return self.get("EVALUATION_CRITERIA", [
            "correctness", "completeness", "coherence", "linguistic_adaptability"
        ])
    
    @property
    def supported_hf_models(self) -> List[str]:
        """Get list of supported Hugging Face models."""
        models = self.get("SUPPORTED_MODELS", {})
        return models.get("hf_models", [])
    
    @property
    def supported_openai_models(self) -> List[str]:
        """Get list of supported OpenAI models."""
        models = self.get("SUPPORTED_MODELS", {})
        return models.get("openai_models", [])
    
    def update_args(self, args) -> None:
        """
        Update argparse arguments with configuration values.
        
        Args:
            args: Argparse arguments object to update
        """
        # Set default paths if not already set
        for attr_name, config_key in [
            ("HF_TOKEN_PATH", "HF_TOKEN_PATH"),
            ("OPENAI_APIKEY_PATH", "OPENAI_APIKEY_PATH"), 
            ("WANDB_TOKEN_PATH", "WANDB_TOKEN_PATH"),
            ("cache_path", "CACHE_PATH"),
            ("hf_model_path", "HF_MODEL_PATH"),
            ("model_save_path", "MODEL_SAVE_PATH"),
            ("quantization", "DEFAULT_QUANTIZATION"),
            ("max_new_tokens", "DEFAULT_MAX_TOKENS"),
            ("dataset", "DEFAULT_DATASET"),
        ]:
            if hasattr(args, attr_name) and getattr(args, attr_name) is None:
                setattr(args, attr_name, self.get(config_key))
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator."""
        return key in self._config
