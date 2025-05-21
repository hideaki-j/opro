#!/usr/bin/env python
"""
Script to analyze optimized instructions from OPRO runs.
Identifies the best performing instructions based on accuracy metrics.
"""
import os
import pandas as pd
import glob
import sys
import argparse
import datetime

def extract_instruction(prompt, instruction_pos):
    """
    Extract instruction from a prompt based on the instruction position.
    
    Args:
        prompt (str): The raw prompt text
        instruction_pos (str): Position of instruction ('Q_begin', 'A_begin', etc.)
    
    Returns:
        str: The extracted instruction or "Unknown" if extraction fails
    """
    try:
        # For GSM8K Q_begin, the instruction is typically at the start of the prompt
        # Example: "Break down the problem step-by-step and compute the answer.\nQ: Natalia sold clips..."
        if instruction_pos == 'Q_begin':
            # First check if there's a Q: in the prompt
            if 'Q:' in prompt:
                # Check if there's text before Q:
                parts = prompt.split('Q:', 1)
                if parts[0].strip():
                    # The instruction is before Q:
                    return parts[0].strip()
                else:
                    # Instruction might be just after Q:
                    instruction_part = parts[1].strip()
                    if '\n' in instruction_part:
                        # Find the instruction before the actual question
                        return instruction_part.split('\n', 1)[0].strip()
                    else:
                        return instruction_part.strip()
            else:
                # No Q: marker, so just take the first line as the instruction
                lines = prompt.strip().split('\n')
                if lines:
                    return lines[0].strip()
        
        elif instruction_pos == 'A_begin':
            # For instructions at the beginning of the answer
            if 'A:' in prompt:
                parts = prompt.split('A:', 1)
                if len(parts) > 1:
                    instruction_part = parts[1].strip()
                    if '\n' in instruction_part:
                        return instruction_part.split('\n', 1)[0].strip()
                    else:
                        return instruction_part.strip()
        
        # If we still haven't found an instruction, try to get it from the filename
        # This is a last resort fallback
        return prompt.strip().split('\n', 1)[0].strip()
    
    except Exception as e:
        print(f"Error extracting instruction: {str(e)}")
        return "Unknown"

def analyze_instructions(result_dir, instruction_pos='Q_begin', top_n=5, output_file=None, csv_output=None):
    """
    Analyze instruction results from CSV files.
    
    Args:
        result_dir (str): Directory containing instruction CSV files
        instruction_pos (str): Position of instruction in prompt
        top_n (int): Number of top instructions to display
        output_file (str): Optional file to save best instruction
        csv_output (str): Optional path to save all results as CSV
    
    Returns:
        list: Sorted list of (instruction, accuracy, filename) tuples
    """
    # Get all CSV files
    csv_files = glob.glob(os.path.join(result_dir, '*.csv'))
    
    print(f"Found {len(csv_files)} CSV files to analyze")
    
    # Store results
    results = []
    
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Calculate average accuracy
            if 'accuracy' in df.columns:
                avg_accuracy = df['accuracy'].mean()
                
                # Try to extract instruction from the prompts
                if not df.empty and 'raw_prompt' in df.columns:
                    first_prompt = df['raw_prompt'].iloc[0]
                    instruction = extract_instruction(first_prompt, instruction_pos)
                    
                    # Store result
                    results.append((instruction, avg_accuracy, os.path.basename(csv_file)))
                else:
                    print(f"Missing required columns in {csv_file}")
            else:
                print(f"No accuracy column in {csv_file}")
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
    
    # Sort by accuracy (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Print the top N instructions
    print(f"\n{'='*60}")
    print(f"TOP {top_n} INSTRUCTIONS BY ACCURACY ON GSM8K")
    print(f"{'='*60}")
    
    for i, (instruction, accuracy, filename) in enumerate(results[:top_n]):
        print(f"{i+1}. INSTRUCTION: \"{instruction}\"")
        print(f"   ACCURACY: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"   FILE: {filename}")
        print(f"   {'='*50}")
    
    # Save best instruction to file if requested
    if output_file and results:
        best_instruction = results[0][0]
        try:
            with open(output_file, 'w') as f:
                f.write(f"{best_instruction}\n")
            print(f"\nBest instruction saved to {output_file}")
        except Exception as e:
            print(f"Error saving to {output_file}: {str(e)}")
    
    # Save all results to CSV if requested
    if csv_output and results:
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(csv_output), exist_ok=True)
            
            # Create DataFrame from results
            df_results = pd.DataFrame(results, columns=['instruction', 'accuracy', 'source_file'])
            
            # Add percentage and rank columns
            df_results['accuracy_percent'] = df_results['accuracy'] * 100
            df_results.insert(0, 'rank', range(1, len(df_results) + 1))
            
            # Save to CSV
            df_results.to_csv(csv_output, index=False, float_format='%.4f')
            print(f"\nAll instructions and accuracies saved to {csv_output}")
        except Exception as e:
            print(f"Error saving CSV to {csv_output}: {str(e)}")
    
    return results

def print_csv_stats(csv_path):
    """
    Print statistics about the saved CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
    """
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found.")
        return
    
    try:
        df = pd.read_csv(csv_path)
        print(f"\nCSV File Statistics:")
        print(f"  - Total instructions analyzed: {len(df)}")
        print(f"  - Average accuracy: {df['accuracy'].mean():.4f} ({df['accuracy'].mean()*100:.2f}%)")
        print(f"  - Best instruction: \"{df.iloc[0]['instruction']}\"")
        print(f"  - Worst instruction: \"{df.iloc[-1]['instruction']}\"")
        print(f"  - Accuracy range: {df['accuracy'].min():.4f} to {df['accuracy'].max():.4f}")
        print(f"  - CSV file saved at: {csv_path}")
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {str(e)}")

def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(description='Analyze OPRO optimization results to find best instructions')
    parser.add_argument('--result_dir', type=str, required=True, 
                        help='Directory containing instruction results (CSV files)')
    parser.add_argument('--instruction_pos', type=str, default='Q_begin', choices=['Q_begin', 'A_begin'],
                        help='Position of the instruction in the prompt')
    parser.add_argument('--top_n', type=int, default=5,
                        help='Number of top instructions to display')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional file to save the best instruction')
    parser.add_argument('--csv_output', type=str, default=None,
                        help='Optional CSV file to save all instructions and accuracies')
    parser.add_argument('--no_timestamp', action='store_true',
                        help='Do not add timestamp to the CSV filename')
    
    args = parser.parse_args()
    
    # Default CSV output path if not specified
    if args.csv_output is None:
        # Create timestamped filename for better organization
        timestamp = "" if args.no_timestamp else f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        csv_path = os.path.join(os.path.dirname(__file__), 'output', f'instruction_results{timestamp}.csv')
    else:
        csv_path = args.csv_output
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    analyze_instructions(
        args.result_dir, 
        instruction_pos=args.instruction_pos,
        top_n=args.top_n,
        output_file=args.output,
        csv_output=csv_path
    )

if __name__ == "__main__":
    main()
