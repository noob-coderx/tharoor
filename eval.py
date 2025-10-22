#!/usr/bin/env python3

import argparse
import json
import math
import sys
import os
from collections import Counter, defaultdict
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import load_jsonl, ensure_dir
from api import FastRewardCalculator

def parse_args():
    """Parse command line arguments for comprehensive evaluation suite.
    
    Configures evaluation parameters for comparing text generation methods across
    multiple metrics including expected reward, perplexity, and entropy.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            Input Configuration:
                - inputs (List[str]): Paths to JSONL prediction files for evaluation
                  Each file evaluated independently for method comparison
                  
            Reward Function Setup:
                - counts_dir (str): Path to llama_test_output directory containing:
                  * trigram_probs.pkl: Pre-computed trigram probabilities
                  * vocab.json: Vocabulary mappings
                  
            Model Configuration:
                - model (str): HuggingFace model ID for teacher-forcing evaluation
                  Used for perplexity calculation via negative log-likelihood
                - hf_token (str): HuggingFace authentication token
                - device (str): PyTorch device for model inference (default: cuda:0)
                
            Output Configuration:
                - out (str): Path for metrics JSON output (default: data/metrics.json)
                  Contains per-file results under method/file keys
                  
    Example:
        python eval.py --inputs outputs_task0_greedy.jsonl outputs_task1_IS.jsonl 
                       --counts-dir /home/daksh/AML/assignment_3/llama_test_output 
                       --model meta-llama/Llama-2-7b-hf --hf-token <token>
    """
    p = argparse.ArgumentParser(description="Evaluate outputs (expected reward, NLL/PPL, entropy).")
    p.add_argument("--inputs", type=str, nargs="+", required=True,
                   help="Prediction files (JSONL). Each file is evaluated independently.")
    p.add_argument("--counts-dir", type=str, default="/home/daksh/AML/assignment_3/llama_test_output", help="Path to llama_test_output/")
    p.add_argument("--model", type=str, required=True, help="HF model id for teacher-forcing (base p0).")
    p.add_argument("--hf-token", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out", type=str, default="data/metrics.json")
    return p.parse_args()

def _load_model(model_name: str, hf_token: str, device: str):
    """Load and configure model and tokenizer for perplexity evaluation.
    
    Args:
        model_name (str): HuggingFace model identifier
        hf_token (str): Authentication token for model access
        device (str): Target device for model placement
        
    Returns:
        Tuple[AutoTokenizer, AutoModelForCausalLM]: Configured tokenizer and model
    """
    # === TOKENIZER LOADING ===
    print(f"\n[DEBUG] Loading model: {model_name}")
    print(f"[DEBUG] Target device: {device}")

    # === Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Check CUDA availability ===
    if torch.cuda.is_available():
        print("[DEBUG] CUDA is available ✅")
        print(f"[DEBUG] Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("[DEBUG] ❌ CUDA not available — will run on CPU. This may be very slow.")

    # === Choose dtype intelligently ===
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 8:  # Ampere or newer (T4/L4/A100 etc.)
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"[DEBUG] Using dtype: {dtype}")

    # === Model ===
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )

    model.eval()
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    # === Post-load sanity check ===
    actual_device = next(model.parameters()).device
    print(f"[DEBUG] Model loaded on: {actual_device}")
    print(f"[DEBUG] EOS token ID: {eos_id}\n")

    # === Optional CUDA performance flags ===
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    return tokenizer, model

def _expected_reward(samples: List[str], weights: List[float], reward_calc: FastRewardCalculator, tok) -> float:
    """Calculate weighted expected reward for generated text samples.
    
    Args:
        samples (List[str]): Generated text samples to evaluate
        weights (List[float]): Importance weights for each sample
                              If empty/mismatched, uniform weights used
        reward_calc (FastRewardCalculator): Trigram-based reward calculator
        tok: Tokenizer for converting text to tokens
        
    Returns:
        float: Weighted expected reward
               - Higher values indicate better quality according to n-gram model
               - Returns 0.0 if no valid samples available
               
    Example:
        >>> samples = ["The cat sat", "Dog runs fast", ""]
        >>> weights = [0.5, 0.3, 0.2]  # Third sample ignored (empty)
        >>> expected_reward = _expected_reward(samples, weights, reward_calc, tok)
    """
    # === INPUT VALIDATION ===
    if not samples:
        return 0.0
    
    # === REWARD COMPUTATION FOR VALID SAMPLES ===
    rewards = []
    valid_indices = []
    
    for i, sample in enumerate(samples):
        # Skip empty or whitespace-only samples
        if sample and sample.strip():
            tokens = tok.tokenize(sample)                      # Llama tokens
            r = float(reward_calc.calculate_reward_tokens(tokens, normalize=True))  # positive
            rewards.append(r)
            valid_indices.append(i)
    
    if not rewards:
        return 0.0  # No valid samples to evaluate
    
    # === WEIGHT NORMALIZATION ===
    if weights and len(weights) == len(samples):
        # Extract weights for valid samples only
        valid_weights = [weights[i] for i in valid_indices]
        weight_sum = sum(valid_weights)
        
        if weight_sum > 0:
            # Normalize weights to sum to 1.0
            normalized_weights = [w / weight_sum for w in valid_weights]
        else:
            # Fallback to uniform if all weights zero
            normalized_weights = [1.0 / len(rewards)] * len(rewards)
    else:
        # Use uniform weights if no weights provided or length mismatch
        normalized_weights = [1.0 / len(rewards)] * len(rewards)
    
    # === EXPECTED REWARD CALCULATION ===
    # E[R] = Σ w̃ᵢ × R(xᵢ)
    expected_reward = sum(w * r for w, r in zip(normalized_weights, rewards))
    return float(expected_reward)

@torch.no_grad()
def _nll_and_ppl(samples: List[str], weights: List[float], tokenizer, model, device: str) -> Dict[str, float]:
    
    if not samples:
        return {"mean_nll": 0.0, "perplexity": 1.0}
    
    sample_nlls = []
    sample_lengths = []
    valid_indices = []
    
    for i, s in enumerate(samples):
        if not s or not s.strip():
            continue
        
        ids = tokenizer.encode(s, add_special_tokens=False, return_tensors="pt").to(device)
        if ids.numel() <= 1:
            continue
        
        # Teacher-forcing NLL calculation
        inputs = ids[:, :-1]
        labels = ids[:, 1:]
        out = model(input_ids=inputs)
        logits = out.logits  # [1, T-1, V]
        logprobs = torch.log_softmax(logits, dim=-1)
        tgt_lp = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
        
        sample_nll = -float(tgt_lp.sum().item())
        sample_nlls.append(sample_nll)
        sample_lengths.append(int(labels.numel()))
        valid_indices.append(i)
    
    if not sample_nlls:
        return {"mean_nll": 0.0, "perplexity": 1.0}
    
    # Use provided weights or uniform weights
    if weights and len(weights) == len(samples):
        # Extract weights for valid samples and normalize
        valid_weights = [weights[i] for i in valid_indices]
        weight_sum = sum(valid_weights)
        if weight_sum > 0:
            normalized_weights = [w / weight_sum for w in valid_weights]
        else:
            normalized_weights = [1.0 / len(sample_nlls)] * len(sample_nlls)
    else:
        # Uniform weights
        normalized_weights = [1.0 / len(sample_nlls)] * len(sample_nlls)
    
    # Weighted average NLL
    weighted_nll = sum(w * nll for w, nll in zip(normalized_weights, sample_nlls))
    
    # Average sequence length for perplexity normalization
    avg_length = sum(w * length for w, length in zip(normalized_weights, sample_lengths))
    
    # Per-token perplexity: exp(NLL / T̄)
    if avg_length > 0:
        ppl = math.exp(weighted_nll / avg_length)
    else:
        ppl = 1.0
    
    return {
        "mean_nll": float(weighted_nll),
        "perplexity": float(ppl),
        "avg_length": float(avg_length)
    }

def _entropy(samples: List[str], weights: List[float], tokenizer) -> float:
    
    if not samples:
        return 0.0
    
    # Count tokens across all samples
    token_counts = defaultdict(float)  # Use defaultdict to handle float weights
    total_tokens = 0.0
    
    # Use provided weights or uniform weights
    # if weights and len(weights) == len(samples):
    #     # Normalize weights
    #     weight_sum = sum(weights)
    #     if weight_sum > 0:
    #         normalized_weights = [w / weight_sum for w in weights]
    #     else:
    #         normalized_weights = [1.0 / len(samples)] * len(samples)
    # else:
    #     # Uniform weights
    #     normalized_weights = [1.0 / len(samples)] * len(samples)
    normalized_weights = [1.0 / len(samples)] * len(samples)
    
    for sample, weight in zip(samples, normalized_weights):
        if sample and sample.strip():
            ids = tokenizer.encode(sample, add_special_tokens=False)
            for token_id in ids:
                token_counts[token_id] += weight
                total_tokens += weight
    
    if total_tokens == 0:
        return 0.0
    
    # Calculate entropy H = -Σ p̂(w) log p̂(w)
    entropy = 0.0
    for count in token_counts.values():
        prob = count / total_tokens
        if prob > 0:
            entropy -= prob * math.log(max(prob, 1e-12))
    
    return float(entropy)

def main():
    args = parse_args()
    ensure_dir("data")

    # Fast reward calculator using trigram probabilities
    cache_file = os.path.join(args.counts_dir, "trigram_probs.pkl")
    if not os.path.exists(cache_file):
        print(f"Warning: trigram_probs.pkl not found at {cache_file}")
        print("Please run: python precompute_trigram_probs.py first")
        sys.exit(1)
    
    reward_calc = FastRewardCalculator(cache_file)

    # Base model for teacher-forcing
    tok, model = _load_model(args.model, args.hf_token, args.device)

    # Evaluate each file independently
    results = {}
    for path in args.inputs:
        print(f"Evaluating {path}...")
        rows = load_jsonl(path)
        
        # Extract all samples and weights
        all_samples = []
        all_weights = []
        methods_seen = set()
        
        for r in rows:
            for block in r.get("continuations", []):
                methods_seen.add(block.get("method", "Unknown"))
                samples = block.get("samples", [])
                
                # Handle different sample formats
                if samples and isinstance(samples[0], dict):
                    # Format: [{"text": "...", "reward": ..., "weight": ...}, ...]
                    texts = [s.get("text", "") for s in samples if isinstance(s, dict)]
                    weights = [s.get("weight", 1.0) for s in samples if isinstance(s, dict)]
                    all_samples.extend(texts)
                    all_weights.extend(weights)
                elif samples and isinstance(samples[0], str):
                    # Format: ["text1", "text2", ...]
                    all_samples.extend(samples)
                    all_weights.extend([1.0] * len(samples))  # Uniform weights
                
                # Also handle normalized_weights if available
                if "normalized_weights" in block and not all_weights:
                    norm_weights = block["normalized_weights"]
                    if len(norm_weights) == len(samples):
                        all_weights = norm_weights

        # Ensure we have weights for all samples
        if len(all_weights) != len(all_samples):
            all_weights = [1.0] * len(all_samples)

        print(f"  Processing {len(all_samples)} samples with {len(all_weights)} weights")

        # Calculate metrics according to assignment specification
        er = _expected_reward(all_samples, all_weights, reward_calc, tok)
        nlp = _nll_and_ppl(all_samples, all_weights, tok, model, args.device)
        H = _entropy(all_samples, all_weights, tok)

        label = f"{path}"
        if len(methods_seen) == 1:
            label = list(methods_seen)[0]
        
        results[label] = {
            "expected_reward": er,
            **nlp,
            "entropy": H,
            "num_samples": len(all_samples)
        }
        
        print(f"  Expected Reward: {er:.4f}")
        print(f"  Perplexity: {nlp['perplexity']:.2f}")
        print(f"  Entropy: {H:.4f}")

    # Save results
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] wrote metrics for {len(args.inputs)} file(s) → {args.out}")

if __name__ == "__main__":
    main()
