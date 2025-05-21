#!/usr/bin/env python
# Copyright 2023 The OPRO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script evaluates an instruction from a file on a test set.
It serves as a simplified wrapper around the evaluate_instructions.py script.

Usage:
python evaluate_instructions_on_test_set.py \
    --instruction_file <path_to_instruction_file> \
    --dataset <dataset_name> \
    --test_split_name <split_name> \
    --scorer_llm_name <scorer_name> \
    --instruction_pos <position> \
    --output_dir <output_directory>
"""

import os
import sys
import datetime
import argparse

# Add the opro root directory to the path
OPRO_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, OPRO_ROOT_PATH)

# Import the relevant modules from the evaluation script
from opro.evaluation import eval_utils
from opro import prompt_utils

try:
    import google.generativeai as palm
except ImportError:
    print("Warning: PaLM API not installed. Cannot use text-bison scorer.")
    
try:
    import openai
except ImportError:
    print("Warning: OpenAI API not installed. Cannot use GPT models as scorers.")

try:
    import sglang
except ImportError:
    print("Warning: SGLang not installed. Using local implementation instead.")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate an instruction from a file on a test set"
    )
    parser.add_argument(
        "--instruction_file", 
        required=True,
        help="Path to a file containing the instruction to evaluate"
    )
    parser.add_argument(
        "--dataset", 
        default="gsm8k",
        choices=["gsm8k", "mmlu", "bbh", "multiarith", "aqua"],
        help="The dataset to evaluate on"
    )
    parser.add_argument(
        "--test_split_name", 
        default="test",
        help="The name of the test split to evaluate on"
    )
    parser.add_argument(
        "--scorer_llm_name", 
        default="sglang",
        choices=["sglang", "text-bison", "gpt-3.5-turbo", "gpt-4"],
        help="The name of the LLM to use for scoring"
    )
    parser.add_argument(
        "--instruction_pos", 
        default="Q_begin",
        choices=["before_Q", "Q_begin", "Q_end", "A_begin"],
        help="The position of the instruction in the prompt"
    )
    parser.add_argument(
        "--output_dir", 
        default="./outputs/test_set_evaluation_results",
        help="Directory to store the evaluation results"
    )
    parser.add_argument(
        "--openai_api_key", 
        default="",
        help="OpenAI API key (only needed for OpenAI models)"
    )
    parser.add_argument(
        "--palm_api_key", 
        default="",
        help="PaLM API key (only needed for text-bison model)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for processing examples (default: 512 for SGLang, 1 for others)"
    )
    
    args = parser.parse_args()
    
    # Read the instruction from the file
    with open(args.instruction_file, 'r') as f:
        instruction = f.read().strip()
    
    print(f"Evaluating instruction:\n{instruction}")
    
    # Setup necessary API keys
    if args.scorer_llm_name in ["gpt-3.5-turbo", "gpt-4"]:
        if not args.openai_api_key:
            args.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not args.openai_api_key:
            raise ValueError("OpenAI API key must be provided for GPT models.")
        openai.api_key = args.openai_api_key
        
    if args.scorer_llm_name == "text-bison":
        if not args.palm_api_key:
            args.palm_api_key = os.environ.get("PALM_API_KEY", "")
        if not args.palm_api_key:
            raise ValueError("PaLM API key must be provided for text-bison model.")
        palm.configure(api_key=args.palm_api_key)
    
    # Setup the scorer function
    if args.scorer_llm_name == "text-bison":
        call_scorer_server_func = prompt_utils.call_palm_server_from_cloud
    elif args.scorer_llm_name in ["gpt-3.5-turbo", "gpt-4"]:
        call_scorer_server_func = lambda prompt: prompt_utils.call_openai_server_func(
            inputs=prompt, 
            model=args.scorer_llm_name, 
            max_decode_steps=1024, 
            temperature=0.0
        )[0]  # Take first element since we're passing a single prompt
    elif args.scorer_llm_name == "sglang":
        from sglang_local.llm import batch_complete # Direct import

        def call_sglang_scorer(batch_of_prompts): # Receives a list from eval_utils
            # Print batch size information for debugging
            print(f"==== BATCH INFO: SGLang received a batch of size {len(batch_of_prompts)} ====")
            
            # Print a sample of the first prompt for debugging
            if batch_of_prompts and len(batch_of_prompts) > 0:
                print(f"First prompt sample (first 100 chars): {batch_of_prompts[0][:100]}...")
                print(f"Last prompt in batch (first 100 chars): {batch_of_prompts[-1][:100]}...")
                print(f"Batch size setting: {batch_size}")
            
            # Directly use batch_complete from sglang_local.llm
            # Pass kwargs like max_tokens and temperature for SGLangWrapper
            try:
                print(f"[DEBUG] Calling batch_complete with batch size {len(batch_of_prompts)}")
                result = batch_complete(
                    prompts=batch_of_prompts,
                    model="sglang", # Ensures SGLangWrapper is used
                    verbose=True, 
                    max_tokens=768, 
                    temperature=0.0
                )
                print(f"==== RESULT: Received {len(result) if result else 'None'} responses ====")
                return result
            except Exception as e:
                print(f"==== SGLANG ERROR: {str(e)} ====")
                raise
        
        call_scorer_server_func = call_sglang_scorer
    else:
        raise ValueError(f"Unsupported scorer: {args.scorer_llm_name}")

    # Setup the dataset and paths
    dataset_name = args.dataset.lower()
    task_name = args.test_split_name.lower()
    
    ROOT_DATA_FOLDER_PATH = os.path.join(OPRO_ROOT_PATH, "data")
    
    if dataset_name == "mmlu":
        root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "MMLU-data")
    elif dataset_name == "bbh":
        root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "BIG-Bench-Hard-data/")
    elif dataset_name == "gsm8k":
        root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "gsm_data")
    elif dataset_name == "aqua":
        root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "AQuA-data")
    else:
        assert dataset_name == "multiarith"
        root_data_folder_path = os.path.join(ROOT_DATA_FOLDER_PATH, "MultiArith-data")
    
    # Create output directory
    datetime_str = (
        str(datetime.datetime.now().replace(microsecond=0))
        .replace(" ", "-")
        .replace(":", "-")
    )
    
    result_folder = os.path.join(
        args.output_dir,
        f"{dataset_name.upper()}-{task_name}-s-{args.scorer_llm_name}-{datetime_str}/",
    )
    
    if not os.path.exists(result_folder):
        os.makedirs(result_folder, exist_ok=True)
    
    print(f"Results will be stored in: {result_folder}")
    
    # Configure dataset-specific settings
    if dataset_name == "mmlu":
        # MMLU specific setup (simplified for this example)
        # You would need to extend this for actual MMLU evaluation
        raise NotImplementedError("MMLU evaluation not yet implemented in this script")
    elif dataset_name == "bbh":
        task_name = args.test_split_name
        raw_data = eval_utils.load_bbh_task_data(task_name, base_dir=root_data_folder_path)
        
        numerical_output_tasks = {
            "object_counting",
            "multistep_arithmetic_two",
        }
        
        boolean_tasks = {
            "boolean_expressions",  # True or False
            "causal_judgement",     # yes or no
            "formal_fallacies",     # valid or invalid
            "navigate",             # yes or no
            "sports_understanding", # yes or no
            "web_of_lies",          # yes or no
        }
        
        multiple_choice_tasks = {
            "date_understanding",
            "disambiguation_qa",
            "geometric_shapes",
            "hyperbaton",
            "logical_deduction_five_objects",
            "logical_deduction_seven_objects",
            "logical_deduction_three_objects",
            "movie_recommendation",
            "penguins_in_a_table",
            "reasoning_about_colored_objects",
            "ruin_names",
            "salient_translation_error_detection",
            "snarks",
            "temporal_sequences",
            "tracking_shuffled_objects_five_objects",
            "tracking_shuffled_objects_seven_objects",
            "tracking_shuffled_objects_three_objects",
        }
        
        prediction_treat_as_number = bool(task_name in numerical_output_tasks)
        prediction_treat_as_bool = bool(task_name in boolean_tasks)
        is_multiple_choice = bool(task_name in multiple_choice_tasks)
        
    elif dataset_name == "gsm8k":
        import pandas as pd
        import numpy as np
        
        # GSM8K specific setup
        task_name = args.test_split_name
        raw_data = pd.DataFrame()
        f_gsm = os.path.join(root_data_folder_path, f"gsm_{task_name}.tsv")
        raw_data = pd.read_csv(f_gsm, sep="\t", header=None)
        
        prediction_treat_as_number = True
        prediction_treat_as_bool = False
        is_multiple_choice = False
        
    elif dataset_name == "aqua":
        # AQuA specific setup
        raw_data = eval_utils.read_jsonl(os.path.join(root_data_folder_path, "AQuA.json"))
        
        prediction_treat_as_number = False
        prediction_treat_as_bool = False
        is_multiple_choice = True
        
    elif dataset_name == "multiarith":
        # MultiArith specific setup
        import json
        with open(os.path.join(root_data_folder_path, "MultiArith.json"), "r") as f:
            raw_data = json.load(f)
            
        prediction_treat_as_number = True
        prediction_treat_as_bool = False
        is_multiple_choice = False
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create the result directory for this specific task
    single_task_result_folder = os.path.join(result_folder, task_name)
    os.makedirs(single_task_result_folder, exist_ok=True)
    
    print(f"\nEvaluating instruction:\n{instruction}")
    print(f"Dataset: {dataset_name.upper()}, Task: {task_name}, Instruction position: {args.instruction_pos}")
    print(f"Prediction as number: {prediction_treat_as_number}, Prediction as boolean: {prediction_treat_as_bool}")
    print(f"Is multiple choice: {is_multiple_choice}")
    
    # Run the evaluation
    if isinstance(raw_data, list) or isinstance(raw_data, dict):
        num_examples = len(raw_data)
    else:  # Assuming it's a DataFrame
        num_examples = raw_data.shape[0]
    
    test_index = list(range(num_examples))
    
    # Set up evaluation parameters
    is_gpt_model = args.scorer_llm_name in ["gpt-3.5-turbo", "gpt-4"]
    
    # Use user-specified batch size or default based on model
    if args.batch_size:
        batch_size = args.batch_size
    else:
        batch_size = 512 if args.scorer_llm_name == "sglang" else 1
    
    # Print evaluation settings for debugging
    print(f"==== EVALUATION SETTINGS ====")
    print(f"Dataset size: {num_examples} examples")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {(num_examples + batch_size - 1) // batch_size}")
    print(f"Scorer model: {args.scorer_llm_name}")
    print(f"Using user-specified batch size: {args.batch_size is not None}")
    print(f"Instruction position: {args.instruction_pos}")
    print(f"==== END SETTINGS ====")
    
    # Use more servers but disable parallel evaluation for stability
    num_servers = 5 if args.scorer_llm_name == "sglang" else 1
    extract_final_answer_by_prompting_again = False
    include_qa = False
    evaluate_in_parallel = False  # Disable parallel evaluation for stability
    
    # Evaluate the instruction
    print(f"[DEBUG] Starting evaluation with batch_size={batch_size}, examples={num_examples}")
    detailed_test_results_df = eval_utils.evaluate_single_instruction(
        data=raw_data,
        instruction=instruction,
        eval_index_all=test_index,  # evaluating all examples
        batch_size=batch_size,
        call_server_func=call_scorer_server_func,
        dataset_name=dataset_name,
        num_servers=num_servers,
        extract_final_answer_by_prompting_again=extract_final_answer_by_prompting_again,
        instruction_pos=args.instruction_pos,
        is_multiple_choice=is_multiple_choice,
        include_qa=include_qa,
        evaluate_in_parallel=evaluate_in_parallel,
        prediction_treat_as_number=prediction_treat_as_number,
        prediction_treat_as_bool=prediction_treat_as_bool,
        prediction_num_decimals=0,
        is_gpt_model=is_gpt_model,
        verbose=True,
        max_retry=5,
        sleep_time=180,
    )
    
    # Save results
    filename = eval_utils.instruction_to_filename(instruction)
    test_file_path = os.path.join(
        single_task_result_folder, f"TEST-{filename}.csv"
    )
    
    print(f"Saving test results to: {test_file_path}")
    detailed_test_results_df.to_csv(test_file_path, index=True, header=True)
    
    # Print summary
    test_scores = detailed_test_results_df["accuracy"]
    print(
        f"Instruction: {instruction}\nAverage test accuracy: {float(test_scores.mean()) * 100:.2f}%"
    )
    
    print(f"\nEvaluation completed. Results saved to: {single_task_result_folder}")

if __name__ == "__main__":
    main()
