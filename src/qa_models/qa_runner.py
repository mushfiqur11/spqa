"""
Question Answering runner for SPQA framework.

This module handles the execution of QA models on style-transferred questions,
providing a unified interface for running different LLMs and collecting their responses.
"""

import os
import argparse
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_from_disk

from ..utils.config import Config
from ..utils.llm_utils import LLMUtils
from ..utils.data_utils import DataUtils


class QARunner:
    """
    Main class for running QA models on style-transferred questions.
    
    This class handles the execution of various LLMs on questions with different
    stylistic variants, collecting answers for evaluation by the LLM-Judge.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize QARunner.
        
        Args:
            config: Configuration object. If None, loads default config.
        """
        self.config = config or Config()
        self.llm_utils = LLMUtils(self.config)
    
    def run_qa_on_dataset(
        self,
        dataset: DatasetDict,
        args,
        output_dir: str,
        style_variants: Optional[List[str]] = None
    ) -> DatasetDict:
        """
        Run QA model on all style variants in the dataset.
        
        Args:
            dataset: Dataset with style-transferred questions
            args: Arguments with model configuration
            output_dir: Output directory for results
            style_variants: List of style variants to process. If None, processes all.
            
        Returns:
            Dataset with generated answers added
        """
        print(f"Running QA with model: {args.model_id}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load LLM for QA
        llm = self.llm_utils.get_llm(args)
        
        # Get style variants to process
        if style_variants is None:
            # Extract style variants from dataset columns
            style_variants = self._extract_style_variants(dataset)
        
        print(f"Processing {len(style_variants)} style variants: {style_variants}")
        
        # Process each split
        processed_dataset = {}
        
        for split_name, split_data in dataset.items():
            print(f"Processing split: {split_name}")
            
            # Add original questions as a variant
            all_variants = ["original"] + style_variants
            
            processed_split = self._process_split_qa(
                split_data, all_variants, llm, args
            )
            
            processed_dataset[split_name] = processed_split
            
            # Save intermediate results for this split
            split_output = os.path.join(output_dir, f"qa_results_{split_name}")
            DataUtils.save_dataset(
                DatasetDict({split_name: processed_split}), 
                split_output, 
                format="both"
            )
        
        final_dataset = DatasetDict(processed_dataset)
        
        # Save complete results
        final_output = os.path.join(output_dir, "complete_qa_results")
        DataUtils.save_dataset(final_dataset, final_output)
        
        print(f"QA execution completed. Results saved to: {output_dir}")
        return final_dataset
    
    def _process_split_qa(
        self,
        split_data: Dataset,
        style_variants: List[str],
        llm,
        args
    ) -> Dataset:
        """
        Process QA for a single dataset split.
        
        Args:
            split_data: Dataset split
            style_variants: Style variants to process (including "original")
            llm: LLM instance
            args: Arguments with configuration
            
        Returns:
            Dataset with QA results added
        """
        # Create a new dataset with expanded samples for each variant
        expanded_samples = []
        
        for sample in tqdm(split_data, desc="Processing samples"):
            for variant in style_variants:
                # Get the question for this variant
                if variant == "original":
                    question = sample.get("question", "")
                    variant_label = "original"
                else:
                    question = sample.get(variant, "")
                    variant_label = variant
                
                if not question.strip():
                    continue  # Skip if no question for this variant
                
                # Generate answer
                generated_answer = self._generate_answer(question, llm, args)
                
                # Create expanded sample
                expanded_sample = {
                    "serial_number": sample.get("serial_number", -1),
                    "original_question": sample.get("question", ""),
                    "gold_answer": sample.get("response", ""),
                    "style_variant": variant_label,
                    "modified_question": question,
                    "generated_answer": generated_answer
                }
                
                expanded_samples.append(expanded_sample)
        
        return Dataset.from_list(expanded_samples)
    
    def _generate_answer(self, question: str, llm, args) -> str:
        """
        Generate an answer for a single question.
        
        Args:
            question: Question to answer
            llm: LLM instance
            args: Arguments with generation parameters
            
        Returns:
            Generated answer
        """
        # Get QA system prompt
        system_prompt = LLMUtils.get_qa_system_prompt()
        
        # Format conversation
        conversation = LLMUtils.format_conversation(question, system_prompt)
        
        try:
            # Generate answer
            answer = llm.generate_response(
                conversation=conversation,
                max_new_tokens=getattr(args, 'max_new_tokens', 200)
            )
            
            return answer.strip()
            
        except Exception as e:
            print(f"Error generating answer for question: {e}")
            return f"Error: Could not generate answer - {str(e)}"
    
    def _extract_style_variants(self, dataset: DatasetDict) -> List[str]:
        """
        Extract style variant column names from dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            List of style variant names
        """
        # Get columns from first split
        first_split = next(iter(dataset.values()))
        all_columns = first_split.column_names
        
        # Filter for style variant columns (exclude standard columns)
        standard_columns = [
            "serial_number", "question", "response", "original_question", 
            "gold_answer", "style_variant", "modified_question", "generated_answer"
        ]
        
        style_variants = [
            col for col in all_columns 
            if col not in standard_columns
        ]
        
        return style_variants
    
    def run_qa_single_model(
        self,
        data_path: str,
        dataset_name: str,
        model_id: str,
        output_dir: str,
        style_variant: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Convenience method to run QA for a single model configuration.
        
        Args:
            data_path: Path to datasets
            dataset_name: Name of dataset to load
            model_id: Model identifier
            output_dir: Output directory
            style_variant: Specific style variant to process (if None, processes all)
            config_path: Path to config file
            **kwargs: Additional arguments
            
        Returns:
            Path to output results
        """
        # Create arguments object
        args = argparse.Namespace(
            model_id=model_id,
            config_path=config_path or self.config.config_path,
            data_path=data_path,
            dataset=dataset_name,
            output_dir=output_dir,
            **kwargs
        )
        
        # Update with config
        self.config.update_args(args)
        
        # Load dataset
        dataset_path = os.path.join(data_path, dataset_name)
        dataset = DataUtils.load_dataset_from_disk(dataset_path)
        
        # Process style variants
        variants_to_process = [style_variant] if style_variant else None
        
        # Run QA
        self.run_qa_on_dataset(
            dataset=dataset,
            args=args,
            output_dir=output_dir,
            style_variants=variants_to_process
        )
        
        return output_dir
    
    def load_qa_results(self, results_path: str) -> DatasetDict:
        """
        Load QA results from disk.
        
        Args:
            results_path: Path to saved QA results
            
        Returns:
            Loaded QA results dataset
        """
        return DataUtils.load_dataset_from_disk(results_path)
    
    def run_batch_qa(
        self,
        dataset: DatasetDict,
        model_configs: List[Dict[str, Any]],
        base_output_dir: str
    ) -> Dict[str, str]:
        """
        Run QA with multiple model configurations.
        
        Args:
            dataset: Dataset with style-transferred questions
            model_configs: List of model configuration dictionaries
            base_output_dir: Base output directory
            
        Returns:
            Dictionary mapping model names to output paths
        """
        results_paths = {}
        
        for i, model_config in enumerate(model_configs):
            model_id = model_config["model_id"]
            print(f"\nRunning QA with model {i+1}/{len(model_configs)}: {model_id}")
            
            # Create model-specific output directory
            model_name_safe = model_id.replace("/", "_")
            model_output_dir = os.path.join(base_output_dir, model_name_safe)
            
            # Create args from config
            args = argparse.Namespace(**model_config)
            self.config.update_args(args)
            
            try:
                # Run QA
                self.run_qa_on_dataset(
                    dataset=dataset,
                    args=args,
                    output_dir=model_output_dir
                )
                
                results_paths[model_id] = model_output_dir
                print(f"Completed QA for {model_id}")
                
            except Exception as e:
                print(f"Error running QA for {model_id}: {e}")
                results_paths[model_id] = f"Error: {str(e)}"
        
        return results_paths
    
    def create_qa_summary(
        self,
        qa_results: DatasetDict,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Create a summary of QA execution results.
        
        Args:
            qa_results: QA results dataset
            output_path: Path to save summary
            
        Returns:
            Summary statistics
        """
        summary = {
            "total_splits": len(qa_results),
            "splits": {},
            "overall_stats": {}
        }
        
        total_samples = 0
        all_variants = set()
        
        # Analyze each split
        for split_name, split_data in qa_results.items():
            split_variants = list(set(split_data["style_variant"]))
            split_stats = {
                "total_samples": len(split_data),
                "style_variants": split_variants,
                "variant_counts": {
                    variant: sum(1 for v in split_data["style_variant"] if v == variant)
                    for variant in split_variants
                }
            }
            
            summary["splits"][split_name] = split_stats
            total_samples += len(split_data)
            all_variants.update(split_variants)
        
        # Overall statistics
        summary["overall_stats"] = {
            "total_samples": total_samples,
            "total_variants": len(all_variants),
            "all_variants": list(all_variants)
        }
        
        # Save summary
        DataUtils.save_json(summary, output_path)
        
        return summary
