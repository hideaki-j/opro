# OPRO Analysis Tools

This directory contains tools for analyzing and extracting insights from OPRO optimization results.

## Instruction Analyzer

The `analyze_instructions.py` script helps you identify the best performing instructions from an OPRO optimization run.

### Features

- Analyzes all instruction CSV files in a results directory
- Calculates average accuracy for each instruction
- Identifies and ranks top-performing instructions
- Supports different instruction positions (Q_begin, A_begin)
- Can save the best instruction to a file
- Exports all instructions with metrics to a CSV file for further analysis
- Provides detailed statistics about the analyzed instructions

### Usage

```bash
# Basic usage
python analyze_instructions.py --result_dir <path_to_result_by_instruction_folder>

# Full options
python analyze_instructions.py \
    --result_dir <path_to_result_by_instruction_folder> \
    --instruction_pos Q_begin \
    --top_n 5 \
    --output best_instruction.txt \
    --csv_output custom_path.csv \
    --no_timestamp
```

### Arguments

- `--result_dir` (required): Path to the `result_by_instruction` folder from an OPRO optimization run
- `--instruction_pos` (optional): Position of the instruction in the prompt (`Q_begin` or `A_begin`, default: `Q_begin`)
- `--top_n` (optional): Number of top instructions to display (default: 5)
- `--output` (optional): File to save the best instruction to (if not specified, no file is saved)
- `--csv_output` (optional): Path to save the CSV results (default: `analysis/output/instruction_results_<timestamp>.csv`)
- `--no_timestamp` (optional): Flag to omit timestamp from the default CSV filename

### Example

```bash
# Analyze results from a specific optimization run
python analysis/analyze_instructions.py \
    --result_dir outputs/optimization-results/GSM8K-train-s-sglang-o-sglang-2025-05-20-22-10-37/result_by_instruction \
    --output best_instruction.txt \
    --top_n 10 \
    --csv_output analysis/output/my_results.csv
```

This will:
1. Analyze all the CSV files in the specified results directory
2. Display the top 5 instructions ranked by accuracy
3. Save the best instruction to `best_instruction.txt`
4. Export all instructions with their metrics to a CSV file in `analysis/output/instruction_results_<timestamp>.csv`

### CSV Output Format

The generated CSV file includes the following columns:

- `rank`: The rank of the instruction based on accuracy (1 = best)
- `instruction`: The extracted instruction text
- `accuracy`: The mean accuracy achieved with this instruction (decimal format)
- `source_file`: The source CSV file containing the detailed results
- `accuracy_percent`: The accuracy as a percentage

This CSV format makes it easy to:

- Compare instructions in spreadsheet software
- Create visualizations of instruction performance
- Conduct further statistical analysis
- Share results with collaborators
- Track performance across optimization runs

## Finding Optimization Results

After running OPRO's `optimize_instructions.py`, results are stored in:

```
outputs/optimization-results/<DATASET>-<TASK>-s-<SCORER>-o-<OPTIMIZER>-<TIMESTAMP>/
```

The `result_by_instruction` subdirectory contains CSV files for each evaluated instruction with detailed performance metrics.

## Instruction Test Set Evaluator

The `evaluate_instructions_on_test_set.py` script helps you evaluate the performance of an optimized instruction on a test set.

### Features

- Evaluates a single instruction on a dataset's test split
- Works with optimized instructions from previous OPRO runs
- Supports different instruction positions (Q_begin, A_begin, etc.)
- Efficiently processes examples in batches
- Saves detailed evaluation results to CSV

### Usage

```bash
# Basic usage
python analysis/evaluate_instructions_on_test_set.py \
    --instruction_file <path_to_instruction_file> \
    --dataset <dataset_name> \
    --test_split_name test \
    --scorer_llm_name <scorer_name> \
    --instruction_pos <position> \
    --output_dir <output_directory>
```

### Arguments

- `--instruction_file` (required): Path to a file containing the instruction to evaluate
- `--dataset` (required): The dataset to evaluate on (one of: gsm8k, mmlu, bbh, multiarith, aqua)
- `--test_split_name` (optional): The name of the test split to evaluate on (default: test)
- `--scorer_llm_name` (optional): The name of the LLM to use for scoring (sglang, text-bison, gpt-3.5-turbo, gpt-4, default: sglang)
- `--instruction_pos` (optional): Position of the instruction in the prompt (before_Q, Q_begin, Q_end, A_begin, default: Q_begin)
- `--output_dir` (optional): Directory to store the evaluation results (default: ./outputs/test_set_evaluation_results)
- `--batch_size` (optional): Number of examples to process in each batch (default: 512 for SGLang, 1 for others). Higher batch sizes significantly improve evaluation speed for SGLang.

### Example

```bash
# Evaluate the best instruction from an optimization run on the GSM8K test set
python analysis/evaluate_instructions_on_test_set.py \
    --instruction_file /projects/0/prjs0808/git/opro/best_instruction.txt \
    --dataset gsm8k \
    --test_split_name test \
    --scorer_llm_name sglang \
    --instruction_pos Q_begin \
    --output_dir ./outputs/test_set_evaluation_results \
    --batch_size 512  # Default is 512 for SGLang, can be adjusted for performance
```

This will:
1. Load the instruction from `best_instruction.txt`
2. Evaluate it on the GSM8K test set
3. Use SGLang as the scorer with efficient batch processing (512 examples per batch)
4. Save detailed results to a timestamped directory in `outputs/test_set_evaluation_results`
5. Display the overall accuracy at the end

### Performance Notes

- The script processes examples in batches to improve performance. For SGLang, it uses a batch size of 512 by default, which significantly improves evaluation speed.
- When using other models like GPT-3.5-turbo or text-bison, the default batch size is 1 (single example processing).
- The batch size parameter can be explicitly set with `--batch_size` to optimize for different hardware or model setups.
- When using SGLang, the script leverages batch processing capabilities to handle multiple prompts efficiently.
- The evaluation results are saved with detailed metrics, allowing for further analysis.

### Workflow

A typical workflow combines both OPRO Analysis tools:

1. Run `analyze_instructions.py` to identify the best instruction from an optimization run and export all results to CSV
2. Review the generated CSV file to compare instruction performance metrics
3. Save the best instruction to a file (e.g., `best_instruction.txt`)
4. Use `evaluate_instructions_on_test_set.py` to evaluate this instruction on the test set
5. Analyze the results to measure the generalization performance of the instruction

## CSV File Statistics

The analyze_instructions.py script automatically prints statistics about the generated CSV file, including:

- Total number of instructions analyzed
- Average accuracy across all instructions
- Best and worst instruction text
- Accuracy range (min to max)
- File path where the CSV was saved

Example output:
```
CSV File Statistics:
  - Total instructions analyzed: 35
  - Average accuracy: 0.6954 (69.54%)
  - Best instruction: "To solve the [problem"
  - Worst instruction: "Let's think carefully to get the right answer."
  - Accuracy range: 0.5123 to 0.8199
  - CSV file saved at: analysis/output/instruction_results_20250521_084512.csv
```

This provides a quick overview of the optimization results without having to open the CSV file.

## Practical Examples

### Comparing Multiple Optimization Runs

You can use the CSV output feature to compare results across different optimization runs:

```bash
# First run
python analysis/analyze_instructions.py \
    --result_dir outputs/optimization-results/Run1/result_by_instruction \
    --csv_output analysis/output/run1_results.csv \
    --no_timestamp

# Second run
python analysis/analyze_instructions.py \
    --result_dir outputs/optimization-results/Run2/result_by_instruction \
    --csv_output analysis/output/run2_results.csv \
    --no_timestamp

# Compare the files using pandas
python -c "import pandas as pd; \
    run1 = pd.read_csv('analysis/output/run1_results.csv'); \
    run2 = pd.read_csv('analysis/output/run2_results.csv'); \
    print('Run1 best accuracy:', run1['accuracy'].max()); \
    print('Run2 best accuracy:', run2['accuracy'].max())"
```

### Quick Usage for Current Optimization Run

To quickly analyze the latest optimization run and generate a CSV:

```bash
# Find the most recent optimization results directory
LATEST_DIR=$(ls -td outputs/optimization-results/*/ | head -1)
RESULT_DIR="${LATEST_DIR}result_by_instruction"

# Analyze and save results
python analysis/analyze_instructions.py \
    --result_dir "$RESULT_DIR" \
    --top_n 10
```

This will automatically save the CSV to `analysis/output/instruction_results_<timestamp>.csv`.

## Test Accuracy Evaluator

The `add_test_accuracy.py` script adds test accuracy metrics to the CSV output from the Instruction Analyzer. This allows you to compare training and test accuracy for instructions.

### Features

- Takes a CSV file with optimized instructions and adds a test accuracy column
- Evaluates each instruction on the test set using `evaluate_instructions_on_test_set.py`
- Maintains the original ranking based on training accuracy
- Adds test accuracy and difference columns to analyze generalization
- Calculates correlation between training and test performance
- Provides comprehensive statistics about instruction performance
- Supports processing only the top N instructions to save time
- Includes a testing mode for development without API calls

### Usage

```bash
# Basic usage
python analysis/add_test_accuracy.py --csv_path analysis/output/instruction_results.csv

# Full options
python analysis/add_test_accuracy.py \
    --csv_path analysis/output/instruction_results.csv \
    --output_path analysis/output/results_with_test.csv \
    --dataset gsm8k \
    --scorer sglang \
    --instruction_pos Q_begin \
    --top_n 5 \
    --delay 2 \
    --batch_size 512 \  # Default is 512 for SGLang, 1 for other models

# List available CSV files in the output directory
python analysis/add_test_accuracy.py --list_csv_files
```

### Arguments

- `--csv_path` (required): Path to the input CSV file with instruction results
- `--output_path` (optional): Path to save the output CSV file (default: automatically generated)
- `--dataset` (optional): Dataset to use for evaluation (default: gsm8k)
- `--scorer` (optional): Model to use for scoring (default: sglang)
- `--instruction_pos` (optional): Position of the instruction in the prompt (default: Q_begin)
- `--top_n` (optional): Only process the top N instructions by training accuracy
- `--delay` (optional): Delay in seconds between API calls to avoid rate limiting
- `--batch_size` (optional): Number of examples to process in each batch (default: 512 for SGLang, 1 for others). Higher values significantly improve throughput with SGLang.
- `--list_csv_files` (optional): List available CSV files in the output directory and exit

### Example

To evaluate only the top 3 instructions from your results CSV:

```bash
python analysis/add_test_accuracy.py \
    --csv_path analysis/output/instruction_results.csv \
    --top_n 3 \
    --scorer sglang \
    --batch_size 512  # Use default batch size of 512 for efficient evaluation
```

This will:
1. Read the CSV file with optimized instructions
2. Process only the top 3 instructions based on training accuracy
3. Evaluate each instruction on the GSM8K test set
4. Add test_accuracy and test_accuracy_percent columns to the CSV
5. Calculate the accuracy difference between training and test
6. Save the results to a new CSV file with "_with_test" suffix
7. Display statistics about the training vs. test accuracy correlation

### Output CSV Columns

The enhanced CSV file will contain the following columns:

- `rank`: Position based on training accuracy (1 = best)
- `instruction`: The optimized instruction text
- `accuracy`: Training accuracy (0-1 scale)
- `source_file`: Original CSV source file
- `accuracy_percent`: Training accuracy as percentage
- `test_accuracy`: Accuracy on the test set (0-1 scale)
- `test_accuracy_percent`: Test accuracy as percentage
- `accuracy_diff`: Difference between training and test accuracy
- `accuracy_diff_percent`: Difference as percentage

## Batch Size Optimization

The batch size parameter (`--batch_size`) significantly impacts evaluation speed:

- **SGLang**: Default batch size is 512, which provides excellent throughput on most systems
  - For very large datasets, you might need to reduce this if you encounter memory issues
  - For systems with more memory, you can try increasing it (e.g., 1024) for even better performance
  
- **Other Models** (GPT-3.5-turbo, text-bison, etc.): Default batch size is 1
  - These API-based models typically process examples one at a time
  - Increasing the batch size usually doesn't improve performance with these models

Example of adjusting batch size for different scenarios:

```bash
# High-performance evaluation with SGLang (large batch)
python analysis/evaluate_instructions_on_test_set.py \
    --instruction_file best_instruction.txt \
    --dataset gsm8k \
    --scorer_llm_name sglang \
    --batch_size 1024

# Memory-constrained environment (smaller batch)
python analysis/evaluate_instructions_on_test_set.py \
    --instruction_file best_instruction.txt \
    --dataset gsm8k \
    --scorer_llm_name sglang \
    --batch_size 128
```
