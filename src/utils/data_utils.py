"""
Data utilities for SPQA framework.

This module provides utilities for loading, saving, and manipulating data
throughout the SPQA pipeline.
"""

import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


class DataUtils:
    """Utility class for data operations in SPQA framework."""
    
    @staticmethod
    def load_dataset_from_json(
        data_files: Union[str, Dict[str, str]], 
        splits: Optional[List[str]] = None
    ) -> DatasetDict:
        """
        Load dataset from JSON files.
        
        Args:
            data_files: Path to JSON file or dict mapping split names to file paths
            splits: List of split names to load (if data_files is a single path)
            
        Returns:
            DatasetDict containing the loaded data
        """
        if isinstance(data_files, str):
            # Single file path provided
            if splits is None:
                splits = ["train"]
            data_files = {split: data_files for split in splits}
        
        return load_dataset("json", data_files=data_files)
    
    @staticmethod
    def load_dataset_from_disk(dataset_path: str) -> DatasetDict:
        """
        Load dataset from disk (saved with save_to_disk).
        
        Args:
            dataset_path: Path to saved dataset
            
        Returns:
            DatasetDict containing the loaded data
        """
        return load_from_disk(dataset_path)
    
    @staticmethod
    def save_dataset(dataset: DatasetDict, output_dir: str, format: str = "both") -> None:
        """
        Save dataset to disk in specified format(s).
        
        Args:
            dataset: DatasetDict to save
            output_dir: Output directory
            format: Format to save in ("disk", "json", or "both")
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if format in ["disk", "both"]:
            # Save as dataset format for efficient loading
            dataset.save_to_disk(output_dir)
        
        if format in ["json", "both"]:
            # Save as JSON files for easy sharing
            for split_name, split_data in dataset.items():
                json_path = os.path.join(output_dir, f"{split_name}.json")
                split_data.to_json(json_path)
    
    @staticmethod
    def load_json(file_path: str) -> Union[Dict, List]:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded JSON data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_json(data: Union[Dict, List], file_path: str) -> None:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            file_path: Output file path
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Loaded data as DataFrame
        """
        return pd.read_csv(file_path)
    
    @staticmethod
    def save_csv(data: pd.DataFrame, file_path: str) -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            data: DataFrame to save
            file_path: Output file path
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data.to_csv(file_path, index=False)
    
    @staticmethod
    def filter_incomplete_samples(dataset: Dataset) -> Dataset:
        """
        Filter out samples with incomplete questions or answers.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Filtered dataset
        """
        def is_complete(sample):
            # Check if question and answer are present and non-empty
            question = sample.get('question', '').strip()
            response = sample.get('response', '').strip()
            return len(question) > 0 and len(response) > 0
        
        return dataset.filter(is_complete)
    
    @staticmethod
    def add_serial_numbers(dataset: Dataset, start_id: int = 0) -> Dataset:
        """
        Add serial numbers to dataset samples.
        
        Args:
            dataset: Input dataset
            start_id: Starting ID number
            
        Returns:
            Dataset with serial numbers added
        """
        def add_id(sample, idx):
            sample['serial_number'] = start_id + idx
            return sample
        
        return dataset.map(add_id, with_indices=True)
    
    @staticmethod
    def combine_style_variants(dataset: DatasetDict, output_dir: str) -> DatasetDict:
        """
        Combine multiple style variant datasets into one.
        
        Args:
            dataset: Dataset with style variants as separate columns
            output_dir: Output directory for combined dataset
            
        Returns:
            Combined dataset with all style variants
        """
        combined_data = {}
        
        for split_name, split_data in dataset.items():
            samples = []
            
            # Create samples for original questions
            for sample in split_data:
                base_sample = {
                    'serial_number': sample['serial_number'],
                    'question': sample['question'],
                    'response': sample['response'],
                    'style_variant': 'original'
                }
                samples.append(base_sample)
            
            # Create samples for each style variant
            style_columns = [col for col in split_data.column_names 
                           if col not in ['serial_number', 'question', 'response']]
            
            for style_col in style_columns:
                for sample in split_data:
                    if sample.get(style_col):  # Check if style variant exists
                        variant_sample = {
                            'serial_number': sample['serial_number'],
                            'question': sample[style_col],
                            'response': sample['response'],
                            'style_variant': style_col
                        }
                        samples.append(variant_sample)
            
            combined_data[split_name] = Dataset.from_list(samples)
        
        combined_dataset = DatasetDict(combined_data)
        
        # Save combined dataset
        DataUtils.save_dataset(combined_dataset, output_dir)
        
        return combined_dataset
    
    @staticmethod
    def validate_qa_format(data: List[Dict]) -> List[str]:
        """
        Validate QA data format and return list of issues found.
        
        Args:
            data: List of QA samples
            
        Returns:
            List of validation issues (empty if no issues)
        """
        issues = []
        required_fields = ['question', 'response']
        
        for i, sample in enumerate(data):
            # Check required fields
            for field in required_fields:
                if field not in sample:
                    issues.append(f"Sample {i}: Missing required field '{field}'")
                elif not isinstance(sample[field], str):
                    issues.append(f"Sample {i}: Field '{field}' must be a string")
                elif not sample[field].strip():
                    issues.append(f"Sample {i}: Field '{field}' is empty")
        
        return issues
    
    @staticmethod
    def create_directory_structure(base_dir: str, subdirs: List[str]) -> None:
        """
        Create directory structure for organizing results.
        
        Args:
            base_dir: Base directory path
            subdirs: List of subdirectories to create
        """
        for subdir in subdirs:
            full_path = os.path.join(base_dir, subdir)
            os.makedirs(full_path, exist_ok=True)
