# generate_task2_smc.py
"""Task 2 — Sequential Monte Carlo (SMC) helpers.
"""
from __future__ import annotations
import os
import math
import torch
from typing import Dict, List, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    raise NotImplementedError("Students must implement this function.")

def cal_intermediate_target_dist(reward_calc: FastRewardCalculator, tokenizer, beta: float, full_ids: List[int]) -> float:
    """
    Args:
        reward_calc: FastRewardCalculator (token_lm.logp available).
        tokenizer: for ids→tokens conversion.
        full_ids: current full context ids (prompt + generated so far).

    Returns:
        float ΔR_t ≥ 0.
    """
    raise NotImplementedError("Students must implement this function.")

@torch.no_grad()
def smc_for_prompt(
    tokenizer: Any,
    model: Any,
    reward_calc: Any,
    *,
    prefix: str,
    N: int,
    max_new_tokens: int,
    eos_id: int,
    beta: float,
    k: int,
) -> Dict:
    """
    Inputs:
      tokenizer, model: HF components from load_model.
      reward_calc: FastRewardCalculator.
      prefix: full prompt string fed to the model (instruction + space + prefix).
      N: number of particles.
      max_new_tokens: continuation budget.
      eos_id: stopping id.
      beta: reward scale.
      k: top-k for proposal.

    Outputs:
      {
        "samples": [ {"text": str, "weight": float}, ... ],   
        "normalized_weights": [float, ...]
      }
    """
    raise NotImplementedError("Students must implement this function.")
