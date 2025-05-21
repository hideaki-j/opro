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
"""Utility functions for calculating training accuracy."""

import numpy as np
import pandas as pd
import sys
import os

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

from opro.evaluation import eval_utils


def calculate_instruction_accuracy(
    data,
    instruction,
    eval_index_all,
    batch_size,
    call_server_func,
    dataset_name,
    num_servers,
    extract_final_answer_by_prompting_again,
    instruction_pos,
    is_multiple_choice,
    include_qa=True,
    evaluate_in_parallel=True,
    num_decodes=1,
    max_retry=5,
    sleep_time=60,
    prediction_treat_as_number=False,
    prediction_treat_as_bool=False,
    prediction_num_decimals=0,
    is_gpt_model=False,
    verbose=False,
):
    """Calculate the accuracy of a given instruction on a dataset.
    
    Args:
        data (list): the input-output pairs.
        instruction (str): the instruction to evaluate.
        eval_index_all (list or np.ndarray): a list of indices to evaluate on.
        batch_size (int): the batch size for model serving.
        call_server_func (function): the function that calls the inference server.
        dataset_name (str): the name of the dataset ("mmlu", "bbh", etc.).
        num_servers (int): the number of inference servers.
        extract_final_answer_by_prompting_again (bool): whether to prompt again.
        instruction_pos (str): where to put the instruction.
        is_multiple_choice (bool or list[bool]): whether questions are multiple choice.
        include_qa (bool): whether to include "Q:" and "A:" formats.
        evaluate_in_parallel (bool): whether to evaluate in parallel.
        num_decodes (int): the number of decodes in model serving.
        max_retry (int): the maximum number of retries.
        sleep_time (int): the number of seconds to sleep before a retry.
        prediction_treat_as_number (bool or 'adaptive'): whether to treat predictions as numbers.
        prediction_treat_as_bool (bool): whether to treat predictions as booleans.
        prediction_num_decimals (int): the number of decimals for numeric predictions.
        is_gpt_model (bool): whether the model is a GPT model.
        verbose (bool): whether to print progress information.
        
    Returns:
        float: The average accuracy across all evaluated examples.
        pandas.DataFrame: The detailed results dataframe.
    """
    # Get the accuracy directly using the return_accuracy_only parameter
    average_accuracy = eval_utils.evaluate_single_instruction(
        data=data,
        instruction=instruction,
        eval_index_all=eval_index_all,
        batch_size=batch_size,
        call_server_func=call_server_func,
        dataset_name=dataset_name,
        num_servers=num_servers,
        extract_final_answer_by_prompting_again=extract_final_answer_by_prompting_again,
        instruction_pos=instruction_pos,
        is_multiple_choice=is_multiple_choice,
        include_qa=include_qa,
        evaluate_in_parallel=evaluate_in_parallel,
        num_decodes=num_decodes,
        max_retry=max_retry,
        sleep_time=sleep_time,
        prediction_treat_as_number=prediction_treat_as_number,
        prediction_treat_as_bool=prediction_treat_as_bool,
        prediction_num_decimals=prediction_num_decimals,
        is_gpt_model=is_gpt_model,
        verbose=verbose,
        return_accuracy_only=True,
    )
    
    # Also get the detailed results if needed
    detailed_results_df = eval_utils.evaluate_single_instruction(
        data=data,
        instruction=instruction,
        eval_index_all=eval_index_all,
        batch_size=batch_size,
        call_server_func=call_server_func,
        dataset_name=dataset_name,
        num_servers=num_servers,
        extract_final_answer_by_prompting_again=extract_final_answer_by_prompting_again,
        instruction_pos=instruction_pos,
        is_multiple_choice=is_multiple_choice,
        include_qa=include_qa,
        evaluate_in_parallel=evaluate_in_parallel,
        num_decodes=num_decodes,
        max_retry=max_retry,
        sleep_time=sleep_time,
        prediction_treat_as_number=prediction_treat_as_number,
        prediction_treat_as_bool=prediction_treat_as_bool,
        prediction_num_decimals=prediction_num_decimals,
        is_gpt_model=is_gpt_model,
        verbose=verbose,
        return_accuracy_only=False,
    )
    
    return average_accuracy, detailed_results_df