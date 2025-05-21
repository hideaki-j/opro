#!/usr/bin/env python
"""
Script to add test accuracy column to instruction CSV results.
This script takes a CSV file with instruction results (from analyze_instructions.py),
evaluates each instruction on a test set, and adds a test_accuracy column.
"""
import os
import sys
import argparse
import pandas as pd
import tempfile
import datetime
import subprocess
import glob
import re
import time
import traceback

# Ensure we can import from parent directory
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

def extract_accuracy_from_output(output_text):
    """
    Extract the accuracy percentage from the evaluation script output.
    
    Args:
        output_text (str): The output text from the evaluation script
        
    Returns:
        float: The test accuracy as a decimal value (0.0-1.0), or None if not found
    """
    # Look for the pattern "Average test accuracy: XX.XX%"
    pattern = r"Average test accuracy: (\d+\.\d+)%"
    match = re.search(pattern, output_text)
    
    if match:
        # Convert percentage to decimal
        accuracy_percent = float(match.group(1))
        return accuracy_percent / 100.0
    
    return None

def get_test_accuracy(instruction, dataset="gsm8k", scorer="sglang", instruction_pos="Q_begin", batch_size=512):
    """
    Run the evaluate_instructions_on_test_set.py script for a single instruction 
    and extract the test accuracy.
    
    Args:
        instruction (str): The instruction to evaluate
        dataset (str): The dataset to use for evaluation
        scorer (str): The model to use for scoring
        instruction_pos (str): The position of the instruction in the prompt
        batch_size (int): Number of examples to process in each batch (default: 512)
        
    Returns:
        float: The test accuracy as a decimal value (0.0-1.0), or None on error
    """
    # Note: Real evaluation will be used in all cases
    
    # Create a temporary file to store the instruction
    instruction_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            instruction_file = f.name
            f.write(instruction)
        
        print(f"Saved instruction to temporary file: {instruction_file}")
        
        # Build the command to run the evaluation script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate_instructions_on_test_set.py")
        
        # Check if the evaluation script exists
        if not os.path.exists(script_path):
            print(f"Error: Evaluation script not found at {script_path}")
            return None
            
        cmd = [
            sys.executable,
            script_path,
            "--instruction_file", instruction_file,
            "--dataset", dataset,
            "--scorer_llm_name", scorer,
            "--instruction_pos", instruction_pos,
            "--batch_size", str(batch_size)
        ]
        
        print(f"Evaluating instruction: \"{instruction[:50]}...\"")
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the evaluation script and capture output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        print(f"[DEBUG] Evaluating with batch_size={batch_size}")
        
        # Set a timeout for the process (e.g., 10 minutes)
        try:
            stdout, stderr = process.communicate(timeout=600)  # 10 minutes timeout
        except subprocess.TimeoutExpired:
            process.kill()
            print("Error: Evaluation process timed out after 10 minutes")
            stdout, stderr = process.communicate()
        
        # Print a summary of the output for debugging
        if stdout:
            print("\nOutput summary:")
            lines = stdout.splitlines()
            if len(lines) > 10:
                # Print the first 5 and last 5 lines
                for line in lines[:5]:
                    print(f"  {line}")
                print("  ...")
                for line in lines[-5:]:
                    print(f"  {line}")
            else:
                print(stdout)
        
        if stderr:
            print("\nErrors:")
            print(stderr)
        
        # Check for errors
        if process.returncode != 0:
            print(f"Error: Evaluation process exited with code {process.returncode}")
            return None
        
        # Extract the accuracy from the output
        accuracy = extract_accuracy_from_output(stdout)
        
        if accuracy is not None:
            print(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            return accuracy
        else:
            print("Warning: Could not extract test accuracy from output")
            return None
    
    except Exception as e:
        print(f"Exception during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Clean up the temporary file
        if instruction_file and os.path.exists(instruction_file):
            try:
                os.unlink(instruction_file)
                print(f"Cleaned up temporary file: {instruction_file}")
            except Exception as e:
                print(f"Warning: Failed to clean up temporary file: {str(e)}")
                pass

def add_test_accuracy_column(csv_path, output_path=None, dataset="gsm8k", 
                            scorer="sglang", instruction_pos="Q_begin", 
                            top_n=None, delay=0, batch_size=512):
    """
    Add a test_accuracy column to a CSV file with instruction results.
    
    Args:
        csv_path (str): Path to the input CSV file
        output_path (str): Path to save the output CSV file (default: automatically generated)
        dataset (str): Dataset to use for evaluation
        scorer (str): Model to use for scoring
        instruction_pos (str): Position of the instruction in the prompt
        top_n (int): Only process the top N instructions by training accuracy
        delay (int): Delay in seconds between API calls to avoid rate limiting
        batch_size (int): Number of examples to process in each batch (default: 512)
        
    Returns:
        str: Path to the output CSV file
    """
    try:
        # Check if the CSV file exists
        if not os.path.exists(csv_path):
            print(f"Error: CSV file {csv_path} does not exist.")
            return None
            
        # Read the input CSV
        df = pd.read_csv(csv_path)
        
        print(f"Successfully loaded CSV with {len(df)} rows and columns: {', '.join(df.columns)}")
        
        # If the CSV already has a rank column, sort by it
        if 'rank' in df.columns:
            print("Using existing rank column for sorting")
            df = df.sort_values('rank')
        else:
            # Otherwise, sort by accuracy (highest first)
            print("Sorting by accuracy and adding rank column")
            df = df.sort_values('accuracy', ascending=False)
            # Add a rank column
            df.insert(0, 'rank', range(1, len(df) + 1))
        
        # If top_n is specified, only keep the top N instructions
        if top_n is not None and top_n > 0:
            original_length = len(df)
            df = df.head(top_n)
            print(f"Processing only the top {top_n} instructions (from {original_length} total)")
        
        # Add a test_accuracy column (initialize with NaN)
        df['test_accuracy'] = float('nan')
        
        # Process each instruction
        for idx, row in df.iterrows():
            instruction = row['instruction']
            
            print(f"\nProcessing instruction {idx+1} of {len(df)}:")
            print(f"Rank: {row['rank'] if 'rank' in df.columns else idx+1}")
            print(f"Training accuracy: {row['accuracy']:.4f} ({row['accuracy']*100:.2f}%)")
            
            
            # Get the test accuracy
            test_accuracy = get_test_accuracy(
                instruction=instruction, 
                dataset=dataset, 
                scorer=scorer, 
                instruction_pos=instruction_pos,
                batch_size=batch_size
            )
            
            # Update the DataFrame
            if test_accuracy is not None:
                df.at[idx, 'test_accuracy'] = test_accuracy
                print(f"Updated test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            else:
                print("Failed to get test accuracy for this instruction")
            
            # Add a delay to avoid rate limiting
            if delay > 0 and idx < len(df) - 1:  # Skip delay after the last instruction
                print(f"Waiting {delay} seconds before next evaluation...")
                time.sleep(delay)
        
        # Generate output path if not specified
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            os.makedirs(output_dir, exist_ok=True)
            
            basename = os.path.basename(csv_path)
            name_parts = os.path.splitext(basename)
            
            output_path = os.path.join(output_dir, f"{name_parts[0]}_with_test{name_parts[1]}")
        
        # Save the updated DataFrame
        df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\nSaved updated CSV to: {output_path}")
        
        # Print some statistics
        test_accuracies = df['test_accuracy'].dropna()
        if not test_accuracies.empty:
            print("\n" + "="*60)
            print(f"STATISTICS:")
            print(f"Total instructions evaluated: {len(test_accuracies)} of {len(df)}")
            print(f"Average test accuracy: {test_accuracies.mean():.4f} ({test_accuracies.mean()*100:.2f}%)")
            print(f"Best test accuracy: {test_accuracies.max():.4f} ({test_accuracies.max()*100:.2f}%)")
            
            # Print the best and worst instructions based on test accuracy
            best_idx = df['test_accuracy'].idxmax()
            print(f"Best instruction by test accuracy: \"{df.loc[best_idx, 'instruction'][:100]}...\"")
            
            # Only calculate correlation if we have enough data
            if len(test_accuracies) > 1:
                correlation = df['accuracy'].corr(df['test_accuracy'])
                print(f"Training vs Test accuracy correlation: {correlation:.4f}")
                
            print(f"Results saved to: {output_path}")
            print("="*60)
        
        return output_path
    
    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(
        description='Add test accuracy column to instruction CSV results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--csv_path', 
        type=str, 
        help='Path to the input CSV file with instruction results'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default=None,
        help='Path to save the output CSV file (default: auto-generated)'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='gsm8k',
        choices=['gsm8k', 'mmlu', 'bbh', 'multiarith', 'aqua'],
        help='Dataset to use for evaluation'
    )
    parser.add_argument(
        '--scorer', 
        type=str, 
        default='sglang',
        choices=['sglang', 'text-bison', 'gpt-3.5-turbo', 'gpt-4'],
        help='Model to use for scoring'
    )
    parser.add_argument(
        '--instruction_pos', 
        type=str, 
        default='Q_begin',
        choices=['Q_begin', 'A_begin', 'before_Q', 'Q_end'],
        help='Position of the instruction in the prompt'
    )
    parser.add_argument(
        '--top_n', 
        type=int, 
        default=None,
        help='Only process the top N instructions by training accuracy'
    )
    parser.add_argument(
        '--delay', 
        type=int, 
        default=0,
        help='Delay in seconds between API calls to avoid rate limiting'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='Number of examples to process in each batch'
    )
    parser.add_argument(
        '--list_csv_files',
        action='store_true',
        help='List available CSV files in the output directory and exit'
    )
    
    args = parser.parse_args()
    
    # If --list_csv_files is specified, print available CSV files and exit
    if args.list_csv_files:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        if os.path.exists(output_dir):
            csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
            if csv_files:
                print("\nAvailable CSV files in output directory:")
                for i, csv_file in enumerate(sorted(csv_files), 1):
                    basename = os.path.basename(csv_file)
                    # Try to read the file to get the number of rows
                    try:
                        df = pd.read_csv(csv_file)
                        row_count = len(df)
                        columns = ', '.join(df.columns)
                        print(f"{i}. {basename} ({row_count} rows)")
                        print(f"   Columns: {columns}")
                        print()
                    except:
                        print(f"{i}. {basename} (Error reading file)")
                print(f"\nTo use one of these files, specify --csv_path {output_dir}/FILENAME.csv")
            else:
                print("No CSV files found in output directory.")
        else:
            print(f"Output directory {output_dir} does not exist.")
        return
    
    # Check if csv_path is provided when not listing files
    if not args.csv_path:
        print("Error: --csv_path is required unless using --list_csv_files")
        parser.print_help()
        sys.exit(1)
    
    # Print configuration
    print("\n" + "="*60)
    print("TEST ACCURACY EVALUATION CONFIGURATION:")
    print(f"Input CSV: {args.csv_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Scorer: {args.scorer}")
    print(f"Instruction position: {args.instruction_pos}")
    if args.top_n:
        print(f"Processing top {args.top_n} instructions")
    if args.delay:
        print(f"Delay between evaluations: {args.delay} seconds")
    print("="*60 + "\n")
    
    # Run the evaluation
    result_path =    add_test_accuracy_column(
        csv_path=args.csv_path,
        output_path=args.output_path,
        dataset=args.dataset,
        scorer=args.scorer,
        instruction_pos=args.instruction_pos,
        top_n=args.top_n,
        delay=args.delay,
        batch_size=args.batch_size
    )
    
    # Print final message
    if result_path:
        print(f"\nEvaluation completed successfully. Results saved to: {result_path}")
    else:
        print("\nEvaluation failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
