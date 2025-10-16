#!/usr/bin/env python3
"""
Task 0: Core Baseline Decoding Implementations

Implemented Algorithms:
    1. Greedy Decoding: Deterministic selection of maximum likelihood tokens
    2. Temperature Sampling: Stochastic sampling with temperature-scaled distributions  
    3. Top-k Sampling: Restricted stochastic sampling from k most probable tokens
"""

from typing import Tuple
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name: str, hf_token: str, device: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, int]:
    """Load and initialize HuggingFace model and tokenizer for baseline decoding.
    
    This function handles the complete model initialization pipeline including tokenizer
    configuration, model loading, device placement, and special token identification.
    Proper setup is critical for consistent baseline performance across all decoding methods.
    
    Args:
        model_name (str): HuggingFace model repository identifier 
            Examples: "meta-llama/Llama-2-7b-hf", "gpt2", "microsoft/DialoGPT-medium"
        hf_token (str): HuggingFace authentication token for accessing gated models
            Required for models like LLaMA, GPT-4, or other restricted access models
        device (str): PyTorch device specification for model placement
            Examples: "cuda:0", "cuda:1", "cpu", "mps" (for Apple Silicon)
    
    Returns:
        Tuple[AutoTokenizer, AutoModelForCausalLM, int]: Model components for generation:
            - tokenizer (AutoTokenizer): Configured tokenizer with proper padding setup
            - model (AutoModelForCausalLM): Model in evaluation mode, placed on specified device  
            - eos_id (int): End-of-sequence token ID for generation termination
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "cuda" in device:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=hf_token,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=hf_token,
            torch_dtype=torch.float32
        )
        model.to("cpu")

    model.eval()
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    return tokenizer, model, eos_id
    

@torch.no_grad()
def greedy_decode(tokenizer, model, prefix: str, max_new: int, eos_id: int) -> str:
    """Perform greedy decoding for deterministic text generation.
            
    Args:
        tokenizer (AutoTokenizer): HuggingFace tokenizer for encoding/decoding
        model (AutoModelForCausalLM): Causal language model in evaluation mode
        prefix (str): Input text prompt to continue
        max_new (int): Maximum number of new tokens to generate
        eos_id (int): End-of-sequence token ID for early termination
        
    Returns:
        str: Generated text continuation (excluding input prefix)
    """
    
    input_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)
    generated = input_ids.clone()
    for _ in range(max_new):
        outputs = model(generated)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        generated = torch.cat([generated, next_token_id.unsqueeze(-1)], dim=-1)
        if next_token_id.item() == eos_id:
            break
    continuation_ids = generated[0, input_ids.shape[1]:]  
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    return continuation_text

@torch.no_grad()
def temperature_decode(tokenizer, model, prefix: str, max_new: int, eos_id: int, tau: float) -> str:
    """Perform temperature sampling for controllable stochastic text generation.
    
    Args:
        tokenizer (AutoTokenizer): HuggingFace tokenizer for encoding/decoding
        model (AutoModelForCausalLM): Causal language model in evaluation mode
        prefix (str): Input text prompt to continue
        max_new (int): Maximum number of new tokens to generate
        eos_id (int): End-of-sequence token ID for early termination
        tau (float): Temperature parameter controlling randomness (must be > 0)
        
    Returns:
        str: Generated text continuation (excluding input prefix)
    """
    input_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)
    generated = input_ids.clone()
    for _ in range(max_new):
        outputs = model(generated)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]  
        max_logit = torch.max(next_token_logits, dim=-1, keepdim=True).values
        shifted_logits = (next_token_logits - max_logit) / tau
        exp_logits = torch.exp(shifted_logits)
        probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
        next_token_id = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token_id], dim=-1)
        if next_token_id.item() == eos_id:
            break
    continuation_ids = generated[0, input_ids.shape[1]:]
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)

    return continuation_text

@torch.no_grad()
def topk_decode(tokenizer, model, prefix: str, max_new: int, eos_id: int, k: int) -> str:
    """Perform top-k sampling for controlled diversity in text generation.

    Args:
        tokenizer (AutoTokenizer): HuggingFace tokenizer for encoding/decoding
        model (AutoModelForCausalLM): Causal language model in evaluation mode
        prefix (str): Input text prompt to continue
        max_new (int): Maximum number of new tokens to generate
        eos_id (int): End-of-sequence token ID for early termination
        k (int): Number of top tokens to consider (must be >= 1)
        
    Returns:
        str: Generated text continuation (excluding input prefix)
    """
    input_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(model.device)
    generated = input_ids.clone()
    for _ in range(max_new):
        outputs = model(generated)
        logits = outputs.logits
        next_token_logits = logits[:, -1, :]  
        topk_values, topk_indices = torch.topk(next_token_logits, k, dim=-1)
        max_logit = topk_values.max(dim=-1, keepdim=True).values
        shifted_topk = topk_values - max_logit
        exp_topk = torch.exp(shifted_topk)
        probs_topk = exp_topk / torch.sum(exp_topk, dim=-1, keepdim=True)
        next_token_id_relative = torch.multinomial(probs_topk, num_samples=1)
        next_token_id = topk_indices.gather(1, next_token_id_relative)
        generated = torch.cat([generated, next_token_id], dim=-1)
        if next_token_id.item() == eos_id:
            break
    continuation_ids = generated[0, input_ids.shape[1]:]
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)

    return continuation_text

