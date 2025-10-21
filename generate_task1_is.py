# generate_task1_is.py
"""Sequential Importance Sampling (Algorithm 1) — generation utilities.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import math
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import the fast trigram API
from api import FastRewardCalculator


def load_counts_and_reward(counts_dir: str, epsilon: float = 1e-9) -> FastRewardCalculator:
    """Initialize trigram-based reward calculator for Sequential Importance Sampling.
    
    Args:
        counts_dir: Directory path containing ngrams data with trigram_probs.pkl cache
        epsilon: Smoothing parameter - minimum probability for unseen trigrams (prevents log(0))
        
    Returns:
        FastRewardCalculator: Configured calculator for computing R(x) rewards
    """
    cache_file = os.path.join(counts_dir, "trigram_probs.pkl")
    return FastRewardCalculator(cache_file, epsilon=epsilon)


def reward_sum_pos_ids(reward_calc: FastRewardCalculator, tokenizer, ids: List[int]) -> float:
    """Compute positive reward on token ids: R_sum over token trigrams.

    Inputs:
        reward_calc: FastRewardCalculator (token_lm.logp available).
        tokenizer: used only to convert ids→tokens.
        ids: full scored context (prompt+continuation) token ids.

    Output:
        R_sum (float). If len(ids) < 3, return 0.0.
    """
    if len(ids) < 3:
        return 0.0

    # Convert ids to tokens
    tokens = tokenizer.convert_ids_to_tokens(ids)
    # Compute reward (already normalized inside FastRewardCalculator)
    return reward_calc.calculate_reward_tokens(tokens, normalize=True)


def load_model(model_name: str, hf_token: str, device: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, int]:
    """Load and configure Hugging Face model components for Sequential Importance Sampling.
    
    Args:
        model_name: Hugging Face model repository ID (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
        hf_token: Authentication token for accessing gated models
        device: Target device for model placement ("cuda:0", "cpu", etc.)
        
    Returns:
        Tuple containing:
            - tokenizer: Configured AutoTokenizer with proper padding token
            - model: AutoModelForCausalLM in evaluation mode on target device
            - eos_id: End-of-sequence token ID for generation termination
    """
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

    return tokenizer, model, eos_id


@torch.no_grad()
def topk_decode_ids(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prefix: str,
    max_new: int,
    k: int,
    eos_id: int,
) -> List[int]:
    """Sample one continuation with top-k proposal. Return continuation token ids (EOS excluded).

    Inputs:
      tokenizer, model: HF components from load_model.
      prefix: full prompt string fed to the model.
      max_new: continuation token budget.
      k: top-k size.
      eos_id: stopping id. Stop early if sampled.

    Output:
      gen_ids: List[int] of sampled token ids for the continuation.
    """
    enc = tokenizer(prefix, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc.input_ids.to(model.device)
    attention_mask = enc.attention_mask.to(model.device)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        top_k=k,
        temperature=1.0,
        max_new_tokens=max_new,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=eos_id,
        use_cache=True
    )[0]

    continuation_ids = output_ids[len(input_ids[0]):].tolist()

    if eos_id in continuation_ids:
        continuation_ids = continuation_ids[:continuation_ids.index(eos_id)]

    return continuation_ids


def importance_sampling_for_prompt(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    reward_calc: FastRewardCalculator,
    *,
    prefix: str,
    K: int,
    max_new_tokens: int,
    eos_id: int,
    beta: float,
    k: int,
) -> Dict:
    """Run SIS for one prompt and return samples with weights.

    Inputs:
      tokenizer, model, reward_calc: initialized components.
      prefix: full prompt string given to the model (instruction + space + prefix).
      K: number of continuations to sample.
      max_new_tokens: continuation budget.
      eos_id: end-of-sequence id.
      beta: reward scale.
      k: top-k for proposal.

    Output dict:
      {
        "samples": [
          {"text": str, "weight": float},
          ...
        ],
        "normalized_weights": [float, ...]   # length K
      }
    """
    samples, weights = [], []

    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids[0].tolist()

    for i in range(K):
        continuation_ids = topk_decode_ids(tokenizer, model, prefix, max_new_tokens, k, eos_id)
        full_ids = prefix_ids + continuation_ids

        R_x = reward_sum_pos_ids(reward_calc, tokenizer, full_ids)
        w = math.exp(beta * R_x)

        text_continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True)
        samples.append({"text": text_continuation, "weight": w})
        weights.append(w)

        if (i + 1) % 5 == 0:
            print(f"[DEBUG] Generated {i+1}/{K} samples ...")

    total_w = sum(weights) if sum(weights) > 0 else 1.0
    normalized_weights = [w / total_w for w in weights]

    return {
        "samples": samples,
        "normalized_weights": normalized_weights
    }
