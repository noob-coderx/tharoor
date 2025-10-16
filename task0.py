#!/usr/bin/env python3
"I wrote some stuff for real"
"""
Task 0: Baseline Decoding Methods for Text Generation

Output Format (JSON Lines):
    Each line contains results for one prompt:
    {
        "prompt_id": int,
        "prefix": str,
        "continuations": [{
            "method": str,           # "Greedy", "Temp-τ", or "TopK-k"
            "samples": [str, ...]    # B generated continuations
        }]
    }
    - transformers: HuggingFace model loading and tokenization

Usage:
    python task0.py --hf-token <token> --method <method>
                    --A <num_prompts> --B <num_samples> [--tau <temp>] [--k <topk>]
    
Note: Uses fixed model meta-llama/Meta-Llama-3-8B-Instruct
"""

import argparse
from utils import set_seed, load_jsonl, save_jsonl, ensure_dir
from generate_task0 import load_model, greedy_decode, temperature_decode, topk_decode

def parse_args():
    """Parse command line arguments for Task 0 baseline decoding methods.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            Model Configuration:
                - hf_token (str): HuggingFace authentication token for model access
                - device (str): CUDA device specification (default: 'cuda:0')
                Note: Uses fixed model meta-llama/Meta-Llama-3-8B-Instruct
                
            Dataset Configuration:
                - test_file (str): Path to JSONL file containing test prompts
                - A (int): Number of prompts to process from dataset
                - B (int): Number of samples to generate per prompt
                - seed (int): Random seed for reproducible results (default: 123)
                
            Decoding Method Selection:
                - method (str): Single method choice from {'greedy', 'temperature', 'topk'}
                
            Method-Specific Parameters:
                - tau (float): Temperature parameter for temperature sampling (default: 1.0)
                - k (int): Top-k parameter for top-k sampling (default: 10)

            Output Configuration:
                - out (str): Output JSONL file path (auto-generated if not provided)
    
    Note:
        Method-specific parameters (tau, k) are only used when the corresponding
        method is selected. Invalid combinations are ignored (e.g., tau with greedy).
    """
    p = argparse.ArgumentParser(description="Task 0: Choose a single decoding method")
    p.add_argument("--hf-token", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--test-file", type=str, default="data/test_prompts.jsonl")
    p.add_argument("--A", type=int, required=True, help="Number of prompts to process")
    p.add_argument("--B", type=int, required=True, help="Number of samples to generate per prompt")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--method", type=str, required=True,
                   choices=["greedy", "temperature", "topk"],
                   help="Which decoding method to run.")
    p.add_argument("--tau", type=float, default=1.0, help="Temperature for --method temperature")
    p.add_argument("--k", type=int, default=10, help="k for --method topk")

    p.add_argument("--out", type=str, default="")
    return p.parse_args()

def _default_out_path(method: str) -> str:
    """Generate default output file path based on selected decoding method.
    
    Args:
        method (str): Decoding method name ('greedy', 'temperature', or 'topk')
        
    Returns:
        str: Formatted output file path in format 'data/outputs_task0_{method}.jsonl'
    """
    return f"data/outputs_task0_{method}.jsonl"

def main():
    """Execute Task 0 baseline decoding across multiple prompts with selected method.
        
    Output Schema (per JSON line):
        {
            "prompt_id": int,              # Original prompt identifier  
            "prefix": str,                 # Input prompt text
            "continuations": [{            # Single method result
                "method": str,             # Method name: "Greedy", "Temp-τ", "TopK-k"
                "samples": [str, ...]      # B generated text continuations
            }]
        }
        
    Raises:
        FileNotFoundError: If test dataset file doesn't exist
        ValueError: If invalid method or hyperparameter values provided
        RuntimeError: If model loading or CUDA operations fail
    """
    # === INITIALIZATION AND SETUP ===
    args = parse_args()
    set_seed(args.seed)  # Ensure reproducible results across runs
    ensure_dir("data")   # Create output directory if needed
    
    # === OUTPUT PATH CONFIGURATION ===
    # Use provided output path or generate default based on method
    out_path = args.out or _default_out_path(args.method)
    print(f"[Task 0] Output will be saved to: {out_path}")

    # === DATASET LOADING ===
    # Load test prompts and limit to requested number A
    rows = load_jsonl(args.test_file)[: args.A]
    print(f"[Task 0] Loaded {len(rows)} prompts from {args.test_file}")

    # === MODEL AND TOKENIZER INITIALIZATION ===
    # Load HuggingFace model, tokenizer, and EOS token ID
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tok, model, eos_id = load_model(model_name, args.hf_token, args.device)
    print(f"[Task 0] Loaded model: {model_name}")
    print(f"[Task 0] Using method: {args.method}")

    # === PROMPT PROCESSING LOOP ===
    out_rows = []
    print(f"[Task 0] Starting {args.method} decoding with B={args.B} samples per prompt")
    
    for row in rows:
        # === PROMPT EXTRACTION ===
        prompt_id = row["prompt_id"]
        prefix = row["prefix"]
        instruction = row.get("instruction", "Continue the text.")
        
        # === FULL PROMPT CONSTRUCTION ===
        # Combine instruction and prefix for complete model input
        full_prompt = f"{instruction} {prefix}"
        max_new = int(row.get("max_output_tokens", 50))
        
        print(f"[Task 0] Processing prompt {prompt_id}: '{prefix[:50]}{'...' if len(prefix) > 50 else ''}'")

        # === METHOD-SPECIFIC SAMPLE GENERATION ===
        samples = []
        
        if args.method == "greedy":
            # === GREEDY DECODING ===
            # Generate B samples (typically B=1 since greedy is deterministic)
            # Note: Multiple samples will be identical for greedy decoding
            for _ in range(args.B):
                sample = greedy_decode(tok, model, full_prompt, max_new, eos_id)
                samples.append(sample)
            method_name = "Greedy"
            
        elif args.method == "temperature":
            # === TEMPERATURE SAMPLING ===
            # Generate B diverse samples using temperature-scaled probabilities
            # Higher τ = more random, lower τ = more focused
            for _ in range(args.B):
                sample = temperature_decode(tok, model, full_prompt, max_new, eos_id, tau=args.tau)
                samples.append(sample)
            method_name = f"Temp-{args.tau}"
            
        elif args.method == "topk":
            # === TOP-K SAMPLING ===
            # Generate B samples from top-k filtered vocabulary
            # k controls diversity: larger k = more diverse samples
            for _ in range(args.B):
                sample = topk_decode(tok, model, full_prompt, max_new, eos_id, k=args.k)
                samples.append(sample)
            method_name = f"TopK-{args.k}"
            
        else:
            # === ERROR HANDLING ===
            # This should not occur due to argparse choices constraint
            method_name = "Unknown"
            print(f"[Task 0 ERROR] Unrecognized method: {args.method}")
            continue

        # === RESULT COMPILATION ===
        # Package results in standardized format for comparison with advanced methods
        out_rows.append({
            "prompt_id": prompt_id,              # Original prompt identifier
            "prefix": prefix,                    # Input prompt text  
            "continuations": [                   # Single method result (Task 0 limitation)
                {
                    "method": method_name,       # Method identifier with parameters
                    "samples": samples           # B generated text continuations
                }
            ]
        })
        
        # === PROGRESS REPORTING ===
        print(f"[Task 0] Completed prompt {prompt_id}: generated {len(samples)} samples")

    # === OUTPUT GENERATION ===
    # Save all results to JSONL file (one JSON object per line)
    save_jsonl(out_path, out_rows)
    
    # === COMPLETION SUMMARY ===
    total_samples = len(out_rows) * args.B
    print(f"\n[Task 0 Complete] Successfully processed {len(out_rows)} prompts")
    print(f"[Task 0 Complete] Generated {total_samples} total samples using {args.method}")
    print(f"[Task 0 Complete] Results saved to: {out_path}")
    
    # === METHOD-SPECIFIC SUMMARY ===
    if args.method == "temperature":
        print(f"[Task 0 Summary] Temperature sampling with τ={args.tau}")
    elif args.method == "topk":
        print(f"[Task 0 Summary] Top-k sampling with k={args.k}")
    elif args.method == "greedy":
        print(f"[Task 0 Summary] Greedy decoding (deterministic)")
    
    print(f"[Task 0 Summary] Model: {model_name}")
    print(f"[Task 0 Summary] Samples per prompt: {args.B}")

if __name__ == "__main__":
    main()
