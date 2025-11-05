"""
Main style transfer module for SPQA framework.

This module implements the Automated Style Transfer (AST) component of the SPQA framework,
which creates linguistic variants of questions while preserving semantic intent.
"""

import argparse
import os
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from datasets import DatasetDict, Dataset

from ..utils.config import Config
from ..utils.llm_utils import LLMUtils
from ..utils.data_utils import DataUtils
from .formality_transfer import FormalityTransfer
from .reading_level_transfer import ReadingLevelTransfer
from .validation import StyleTransferValidator


class StyleTransfer:
    """
    Main class for orchestrating style transfer operations across multiple dimensions.
    
    This class handles the creation of stylistically varied question versions while
    preserving semantic meaning, as described in the SPQA paper.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize StyleTransfer.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or Config()
        self.llm_utils = LLMUtils(self.config)
        
        # Initialize specialized transfer modules
        self.formality_transfer = FormalityTransfer(self.config)
        self.reading_level_transfer = ReadingLevelTransfer(self.config)
        self.validator = StyleTransferValidator(self.config)
    
    def transfer_single_sample(
        self, 
        question: str, 
        style_type: str, 
        style_variant: str, 
        llm,
        max_new_tokens: int = 200
    ) -> str:
        """
        Transfer a single question to a specific style variant.
        
        Args:
            question: Original question text
            style_type: Type of style transfer (formality, reading_level, domain_knowledge)
            style_variant: Specific variant within the style type
            llm: LLM instance for generation
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Style-transferred question
        """
        # Get appropriate system prompt
        system_prompt = LLMUtils.get_style_transfer_prompt(style_type, style_variant)
        
        # Format conversation
        conversation = LLMUtils.format_conversation(question, system_prompt)
        
        # Generate response
        return llm.generate_response(
            conversation=conversation, 
            max_new_tokens=max_new_tokens
        )
    
    def apply_formality_transfer(
        self, 
        dataset: DatasetDict, 
        args,
        output_dir: str
    ) -> DatasetDict:
        """
        Apply formality-based style transfers to dataset.
        
        Args:
            dataset: Input dataset
            args: Arguments with model configuration
            output_dir: Output directory for results
            
        Returns:
            Dataset with formality variants added
        """
        return self.formality_transfer.apply_transfer(dataset, args, output_dir)
    
    def apply_reading_level_transfer(
        self, 
        dataset: DatasetDict, 
        args,
        output_dir: str
    ) -> DatasetDict:
        """
        Apply reading level-based style transfers to dataset.
        
        Args:
            dataset: Input dataset
            args: Arguments with model configuration  
            output_dir: Output directory for results
            
        Returns:
            Dataset with reading level variants added
        """
        return self.reading_level_transfer.apply_transfer(dataset, args, output_dir)
    
    def apply_all_transfers(
        self,
        dataset: DatasetDict,
        args,
        output_dir: str,
        validate: bool = True
    ) -> DatasetDict:
        """
        Apply all style transfers to the dataset.
        
        Args:
            dataset: Input dataset
            args: Arguments with model configuration
            output_dir: Output directory for results
            validate: Whether to validate transfers
            
        Returns:
            Dataset with all style variants added
        """
        print("Starting comprehensive style transfer...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load LLM for style transfer
        llm = self.llm_utils.get_llm(args)
        
        # Track all variants to add
        all_variants = []
        all_variants.extend(self.config.style_variants["formality"])
        all_variants.extend(self.config.style_variants["reading_level"])
        all_variants.extend(self.config.style_variants["domain_knowledge"])
        
        print(f"Applying {len(all_variants)} style variants: {all_variants}")
        
        # Apply each style variant
        for variant in tqdm(all_variants, desc="Style variants"):
            # Determine style type
            if variant in self.config.style_variants["formality"]:
                style_type = "formality"
            elif variant in self.config.style_variants["reading_level"]:
                style_type = "reading_level"
            else:
                style_type = "domain_knowledge"
            
            # Create temporary args for this variant
            temp_args = argparse.Namespace(**vars(args))
            temp_args.style_variant = variant
            temp_args.style_type = style_type
            
            # Apply transfer
            dataset = dataset.map(
                lambda sample: self._transfer_sample(sample, style_type, variant, llm, args),
                desc=f"Applying {variant} style"
            )
            
            # Save intermediate results
            intermediate_output = os.path.join(output_dir, f"intermediate_{variant}")
            DataUtils.save_dataset(dataset, intermediate_output, format="disk")
        
        # Save final combined dataset
        final_output = os.path.join(output_dir, "all_variants")
        DataUtils.save_dataset(dataset, final_output)
        
        # Validate transfers if requested
        if validate:
            print("Validating style transfers...")
            validation_results = self.validator.validate_dataset(dataset, all_variants)
            
            # Save validation results
            validation_output = os.path.join(output_dir, "validation_results.json")
            DataUtils.save_json(validation_results, validation_output)
            
            print(f"Validation results saved to: {validation_output}")
        
        print(f"Style transfer complete. Results saved to: {output_dir}")
        return dataset
    
    def _transfer_sample(
        self, 
        sample: Dict[str, Any], 
        style_type: str, 
        style_variant: str, 
        llm,
        args
    ) -> Dict[str, Any]:
        """
        Internal method to transfer a single sample.
        
        Args:
            sample: Dataset sample
            style_type: Type of style transfer
            style_variant: Specific variant
            llm: LLM instance
            args: Arguments with configuration
            
        Returns:
            Sample with style variant added
        """
        original_question = sample["question"]
        
        try:
            transferred_question = self.transfer_single_sample(
                original_question, 
                style_type, 
                style_variant, 
                llm,
                getattr(args, 'max_new_tokens', 200)
            )
            sample[style_variant] = transferred_question
        except Exception as e:
            print(f"Error transferring sample to {style_variant}: {e}")
            sample[style_variant] = original_question  # Fallback to original
        
        return sample
    
    def load_dataset(
        self, 
        dataset_name: str,
        data_path: str,
        splits: Optional[List[str]] = None
    ) -> DatasetDict:
        """
        Load dataset for style transfer.
        
        Args:
            dataset_name: Name of the dataset
            data_path: Path to dataset files
            splits: Specific splits to load
            
        Returns:
            Loaded dataset
        """
        if splits is None:
            splits = ["QATRAIN"]  # Default for MedRedQA format
        
        dataset_dir = os.path.join(data_path, dataset_name)
        
        # Try to load as saved dataset first
        try:
            return DataUtils.load_dataset_from_disk(dataset_dir)
        except:
            # Load from JSON files
            data_files = {}
            for split in splits:
                json_file = os.path.join(dataset_dir, f"{split}.json")
                if os.path.exists(json_file):
                    data_files[split] = json_file
            
            if not data_files:
                raise FileNotFoundError(f"No dataset files found in {dataset_dir}")
            
            return DataUtils.load_dataset_from_json(data_files)
    
    def preprocess_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """
        Preprocess dataset by filtering incomplete samples and adding metadata.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Preprocessed dataset
        """
        processed_dataset = {}
        
        for split_name, split_data in dataset.items():
            # Filter incomplete samples
            filtered_data = DataUtils.filter_incomplete_samples(split_data)
            
            # Add serial numbers
            processed_data = DataUtils.add_serial_numbers(filtered_data)
            
            processed_dataset[split_name] = processed_data
            
            print(f"Split '{split_name}': {len(split_data)} -> {len(processed_data)} samples after preprocessing")
        
        return DatasetDict(processed_dataset)
