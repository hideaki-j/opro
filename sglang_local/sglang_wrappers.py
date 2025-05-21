import time
import itertools
import requests
import concurrent.futures
from tqdm import tqdm
from transformers import AutoTokenizer
import copy
import psutil

def sanity_check_if_model_is_running(llm_model_name: str) -> bool:
    """
    Check if the model is currently running in any sglang server process.
    Returns True if the model is running, False otherwise.
    """  
    # Extract the model name after the last '/'
    _model_name = llm_model_name.split('/')[-1]
    
    # Track unique running models
    running_models = set()
    
    # Check all running processes
    for proc in psutil.process_iter(['cmdline', 'pid']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline:  # Check if cmdline exists and is not empty
                cmdline_str = ' '.join(cmdline)
                # If it's a sglang server process, print the model info
                if 'sglang.launch_server' in cmdline_str:
                    # Extract model path from command line
                    cmd_parts = cmdline_str.split()
                    for i, part in enumerate(cmd_parts):
                        if part == '--model-path' and i + 1 < len(cmd_parts):
                            running_model = cmd_parts[i + 1]
                            running_models.add(running_model)
                            # Check if this is the model we're looking for
                            if _model_name in running_model:
                                print(f"Model {llm_model_name} is running")
                                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    # Include available models in the exception message
    error_msg = f"Model {llm_model_name} is not running. "
    if running_models:
        error_msg += "Available running models: " + ", ".join(f"{model}" for model in running_models)
    else:
        error_msg += "No sglang models currently running"
    
    raise Exception(error_msg)


class PromptFormatter:
    def __init__(self, sys_prompt=None, llm_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
        sanity_check_if_model_is_running(llm_model_name)
        if "gemma" in llm_model_name:
            assert sys_prompt is None, "System prompt is not supported for Gemma models"
        if sys_prompt is not None:
            self.sys_prompt = sys_prompt
        else:
            self.sys_prompt = "You are a professional assistant who strictly follows the instructions."
        self.llm_model_name = llm_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)

    def format_prompt(self, prompt: str) -> list:
        if "gemma" in self.llm_model_name:
            messages = [
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": prompt},
            ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def check_if_already_formatted(self, prompt: str) -> bool:
        if "Llama-3" in self.llm_model_name:
            return prompt.strip().startswith("<|begin_of_text|>")
        elif "gemma" in self.llm_model_name:
            return prompt.strip().startswith("<bos><start_of_turn>")
        elif "Qwen" in self.llm_model_name or "Athene" in self.llm_model_name or "SmolLM2" in self.llm_model_name:
            return prompt.strip().startswith("<|im_start|>")
        # elif "Phi-" in self.llm_model_name: # Phi does not work on SGLang properly; need time to fix
        #     return prompt.strip().startswith("<|system|>") or prompt.strip().startswith("<|user|>")
        elif "Ministral" in self.llm_model_name:
            return prompt.strip().startswith("<s>")
        else:
            raise ValueError(f"Unsupported model: {self.llm_model_name}")

class SGLangWrapper:
    def __init__(self, max_tokens=768, temperature=0.0, sys_prompt=None, specify_ports=None, llm_model_name='meta-llama/Meta-Llama-3.1-8B-Instruct'):
        
        self.max_walkers_per_server = 500
        self.llm_model_name = llm_model_name
        sanity_check_if_model_is_running(llm_model_name)

        if specify_ports is None or (len(specify_ports) == 1 and specify_ports[0] == None): # None or [None]
            self.urls = [
                "http://localhost:31011/generate",
                "http://localhost:31022/generate",
                "http://localhost:31033/generate",
                "http://localhost:31044/generate",
                "http://localhost:31055/generate",
                "http://localhost:31066/generate",
                "http://localhost:31077/generate",
                "http://localhost:31088/generate",
                "http://localhost:31099/generate",
                "http://localhost:31000/generate",
            ]
        else:
            self.urls = [f'http://localhost:{port}/generate' for port in specify_ports]
        self.headers = {
            "Content-Type": "application/json"
        }
        self.sampling_params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature
        }
        if temperature == 0.0:
            print('TODO: Even though the temperature is set to 0.0, the model may still generate text with some randomness. Need to investigate further.')
        self.formatter = PromptFormatter(sys_prompt=sys_prompt, llm_model_name=llm_model_name)

    def _check_url_availability(self, url) -> bool:
        test_data = {
            "text": "test",
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0
            }
        }
        try:
            response = requests.post(url, headers=self.headers, json=test_data)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def check_port_availability(self, port) -> bool:
        """This is used in slurm jobs to check server is ready."""
        url = f"http://localhost:{port}/generate"
        return self._check_url_availability(url)

    def set_available_urls(self):
        available_urls = []
        for url in self.urls:
            if self._check_url_availability(url):
                print(f"URL {url} is available")
                available_urls.append(url)
            else:
                print(f"URL {url} is NOT available")
                continue
        print(f"Available URLs: {available_urls}")
        self.available_urls = available_urls

    def send_request(self, url, prompt):
        data = {
            "text": prompt,
            "sampling_params": self.sampling_params
        }
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code} - {response.text}"
        
    def apply_prompt_formatting(self, arg_prompts):
        if not self.formatter.check_if_already_formatted(arg_prompts[0]):
            print("Applying prompt formatting")
            prompts = [self.formatter.format_prompt(prompt) for prompt in arg_prompts]
        else:
            print("Input prompts are already formatted, skipping prompt formatting")
            prompts = arg_prompts
        return prompts
    
    def _print_results(self, results, input_tokens_per_second, output_tokens_per_second):
        for idx, result in enumerate(results):
            print(f"Result {idx}: {result}")

        print(f"Input tokens per second: {input_tokens_per_second:.2f}")
        print(f"Output tokens per second: {output_tokens_per_second:.2f}")

    def _get_completion_from_prompts(self, prompts, verbose=False) -> list:
        assert isinstance(prompts, list), "Prompts must be a list of strings"
        self.set_available_urls()
        prompts = self.apply_prompt_formatting(prompts)
        results = [None] * len(prompts)
        start_time = time.time()
        
        # Distribute requests among available URLs
        url_cycle = itertools.cycle(self.available_urls)
        futures = {}
        max_workers = self.max_walkers_per_server * len(self.available_urls)
        total_prompt_tokens = 0
        total_completion_tokens = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, prompt in enumerate(prompts):
                url = next(url_cycle)
                futures[executor.submit(self.send_request, url, prompt)] = idx
                
            with tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing results") as pbar:
                for future in pbar:
                    result_idx = futures[future]
                    result = future.result()
                    results[result_idx] = result  # store the result in the original order
                    
                    if isinstance(result, dict):
                        total_prompt_tokens += result['meta_info']['prompt_tokens']
                        total_completion_tokens += result['meta_info']['completion_tokens']
                    
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        input_tokens_per_second = total_prompt_tokens / elapsed_time
                        output_tokens_per_second = total_completion_tokens / elapsed_time
                        pbar.set_postfix_str(
                            f"est. speed input: {input_tokens_per_second:.2f} toks/s, output: {output_tokens_per_second:.2f} toks/s"
                        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        print(f"Total prompt tokens: {total_prompt_tokens}")
        print(f"Total completion tokens: {total_completion_tokens}")

        input_tokens_per_second = total_prompt_tokens / elapsed_time
        output_tokens_per_second = total_completion_tokens / elapsed_time

        if verbose:
            self._print_results(results, input_tokens_per_second, output_tokens_per_second)
        
        # convert results to a list of completion texts
        results = [result['text'] for result in results if isinstance(result, dict)]

        for i, (prompt, response) in enumerate(zip(prompts, results)):
            if i >= 3:
                print(f"Only showing first 3 results")
                break
            print(f"========= PROMPT {i} =========")
            print(f"\033[92m{prompt}\033[00m")
            print(f"\033[94m{response}\033[00m")
            print()
            
        return results
    
    def get_completion_from_prompts(self, prompts: list, verbose=False) -> list:
        max_retry = 5
        for i in range(max_retry):
            try:
                return self._get_completion_from_prompts(prompts, verbose)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Retry {i+1}/{max_retry}")
                continue
        raise Exception("Failed to get completions after multiple retries")
    
    def add_valid_completions(self, prompts_with_check_functions: list, max_retries: int, verbose=False) -> list:
        """
        prompts_with_check_functions: list of dict
        - 'prompt': str
        - 'check_function': function that takes in a completion and returns a boolean
            The function should be like `completion_check_function(completion) -> bool`
            NOTE: Use lambda to set a variable check_function, 
                e.g. `check_function = lambda completion: completion_check_function(completion, var=dummy_var)`
        
        Returns:
            list: List of original dictionaries with added 'completion' key. Failed validations will have completion=None.
        """
        assert isinstance(prompts_with_check_functions, list), "prompts_with_check_functions must be a list"
        required_keys = {'prompt', 'check_function'}
        assert all(required_keys.issubset(d.keys()) for d in prompts_with_check_functions), "Each dict must contain 'prompt' and 'check_function' keys"
        assert max_retries > 0, "max_retries must be positive"

        # Create a deep copy to avoid modifying the input
        result_dicts = copy.deepcopy(prompts_with_check_functions)
        for d in result_dicts:
            d['completion'] = None

        incomplete_indices = list(range(len(prompts_with_check_functions)))
        
        for i_retry in range(max_retries):
            print(f"Retry {i_retry+1}/{max_retries}")
            if not incomplete_indices:
                break
                
            retry_prompts = [prompts_with_check_functions[i]['prompt'] for i in incomplete_indices]
            completions = self.get_completion_from_prompts(retry_prompts)
            
            for idx_in_batch, completion in enumerate(completions):
                original_idx = incomplete_indices[idx_in_batch]
                current_check_function = prompts_with_check_functions[original_idx]['check_function']
                if current_check_function(completion):
                    result_dicts[original_idx]['completion'] = completion
            
            # Update incomplete indices
            incomplete_indices = [i for i, d in enumerate(result_dicts) if d['completion'] is None]
        
        if incomplete_indices and verbose:
            print(f"Warning: {len(incomplete_indices)} prompts failed validation after {max_retries} retries")
        
        return result_dicts