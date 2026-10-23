"""
GCG Utility Functions for Defensive Suffix Generation

This module contains utility functions for:
- Loss computation (cross-entropy for target refusal phrases)
- Gradient computation for suffix tokens
- Candidate token sampling based on gradients
- Token filtering and validation
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import logging
import gc

logger = logging.getLogger(__name__)


def compute_refusal_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    target_slice: slice,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute cross-entropy loss for target refusal phrase.

    This is the OPPOSITE of jailbreak GCG:
    - Jailbreak GCG minimizes loss for "Sure, here is..."
    - Defensive GCG minimizes loss for "Sorry, I can't..."

    Args:
        model: The language model
        input_ids: Full input token IDs [batch_size, seq_len]
        target_ids: Target refusal phrase token IDs [target_len]
        target_slice: Slice indicating where target tokens should appear in output
        attention_mask: Optional attention mask

    Returns:
        Loss tensor (scalar)
    """
    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [batch_size, seq_len, vocab_size]

    # Get logits for the target positions (where we want refusal to appear)
    # Target starts after the input, so we look at logits shifted by 1
    target_logits = logits[:, target_slice, :]  # [batch_size, target_len, vocab_size]

    # Reshape for cross-entropy
    # target_logits: [batch_size * target_len, vocab_size]
    # target_ids: [batch_size * target_len]
    batch_size = input_ids.shape[0]
    target_len = target_ids.shape[0]

    target_logits_flat = target_logits.reshape(-1, target_logits.shape[-1])
    target_ids_expanded = target_ids.unsqueeze(0).expand(batch_size, -1).reshape(-1)

    # Cross-entropy loss
    loss = F.cross_entropy(target_logits_flat, target_ids_expanded, reduction='mean')

    return loss


def get_token_gradients(
    model: torch.nn.Module,
    embedding_layer: torch.nn.Module,
    input_ids: torch.Tensor,
    suffix_slice: slice,
    target_ids: torch.Tensor,
    target_slice: slice,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Compute gradients with respect to suffix token embeddings.

    Handles multi-GPU scenarios where input and output devices may differ.

    Args:
        model: The language model
        embedding_layer: The token embedding layer (model.model.embed_tokens for Llama)
        input_ids: Full input token IDs [batch_size, seq_len]
        suffix_slice: Slice indicating suffix token positions
        target_ids: Target refusal phrase token IDs
        target_slice: Slice for target positions in output
        attention_mask: Optional attention mask

    Returns:
        Tuple of (gradients [suffix_len, hidden_dim], loss value)
    """
    # Determine input device from embedding layer
    input_device = next(embedding_layer.parameters()).device

    # Ensure input_ids are on the correct device
    input_ids = input_ids.to(input_device)

    # Get embeddings - this determines the input device.
    # .detach() is required: embedding_layer(input_ids) is a non-leaf tensor
    # (produced by the embedding op on the weight leaf).  Non-leaf tensors do
    # not accumulate .grad during backward.  Detaching first makes the cloned
    # tensor a true leaf, so .grad is populated after loss.backward().
    embeds = embedding_layer(input_ids).detach()   # [batch_size, seq_len, hidden_dim]
    embeds = embeds.clone().requires_grad_(True)   # leaf tensor — .grad will be filled

    # Only compute logits for target positions to avoid materialising a
    # [B, seq_len, vocab] tensor.  For Llama-3-8B with seq_len≈600 and
    # vocab=128256 that would be ~7.3 GB per batch even at B=1.
    #
    # num_logits_to_keep=N (transformers ≥4.43) tells lm_head to only process
    # the last N hidden states, giving a [B, N, vocab] tensor instead.
    # The target slice sits at positions [seq_len-N .. seq_len-1] in the
    # full sequence, so it maps to [0 .. len(target_ids)-1] in the kept tensor.
    seq_len = input_ids.shape[1]
    num_logits_to_keep = seq_len - target_slice.start   # = len(target_ids) + 1

    # Forward pass — only materialise logits at target positions.
    # transformers ≥5.x renamed the kwarg from num_logits_to_keep → logits_to_keep.
    # use_cache=False: critical for large models — otherwise KV cache (~hundreds
    # of MB per call) accumulates and OOMs the allocator after a few steps.
    try:
        outputs = model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            logits_to_keep=num_logits_to_keep,
            use_cache=False,
        )
    except TypeError:
        try:
            outputs = model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                num_logits_to_keep=num_logits_to_keep,
                use_cache=False,
            )
        except TypeError:
            outputs = model(inputs_embeds=embeds, attention_mask=attention_mask, use_cache=False)

    logits = outputs.logits   # [B, num_logits_to_keep, vocab] OR [B, seq_len, vocab]

    # Determine target slice based on actual logits shape.
    # If num_logits_to_keep was honoured, logits.shape[1] < seq_len and the
    # target sits at the START of the kept tensor.
    # If the param was silently ignored, logits.shape[1] == seq_len and we
    # use the original target_slice directly.
    if logits.shape[1] < seq_len:
        adj_target_slice = slice(0, target_slice.stop - target_slice.start)
    else:
        adj_target_slice = target_slice

    # Get the device where logits are (output device, may differ in multi-GPU)
    output_device = logits.device

    # Compute loss
    batch_size = input_ids.shape[0]
    target_logits = logits[:, adj_target_slice, :]
    target_logits_flat = target_logits.reshape(-1, target_logits.shape[-1])
    target_ids_expanded = target_ids.to(output_device).unsqueeze(0).expand(batch_size, -1).reshape(-1)

    loss = F.cross_entropy(target_logits_flat, target_ids_expanded, reduction='mean')

    # Backward pass
    loss.backward()

    # Get gradients for suffix positions (on input device where embeddings live)
    suffix_grads = embeds.grad[:, suffix_slice, :]  # [batch_size, suffix_len, hidden_dim]

    # Average over batch and ensure on input device
    avg_grads = suffix_grads.mean(dim=0).to(input_device)  # [suffix_len, hidden_dim]

    # Clean up to free memory
    loss_value = loss.item()
    del outputs, logits, target_logits, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return avg_grads, loss_value


def compute_candidates_loss(
    model: torch.nn.Module,
    embedding_layer: torch.nn.Module,
    input_ids: torch.Tensor,
    suffix_slice: slice,
    candidate_ids: torch.Tensor,
    target_ids: torch.Tensor,
    target_slice: slice,
    attention_mask: Optional[torch.Tensor] = None,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Evaluate multiple candidate suffixes and return their losses.

    Args:
        model: The language model
        embedding_layer: The token embedding layer
        input_ids: Base input token IDs [1, seq_len]
        suffix_slice: Slice indicating suffix positions
        candidate_ids: Candidate suffix token IDs [num_candidates, suffix_len]
        target_ids: Target refusal phrase token IDs
        target_slice: Slice for target positions
        attention_mask: Optional attention mask
        batch_size: Batch size for evaluation

    Returns:
        Losses for each candidate [num_candidates]
    """
    # Get input device from input_ids
    input_device = input_ids.device
    candidate_ids = candidate_ids.to(input_device)

    num_candidates = candidate_ids.shape[0]
    all_losses = []

    # Process in batches to avoid OOM
    for i in range(0, num_candidates, batch_size):
        batch_candidates = candidate_ids[i:i + batch_size]
        curr_batch_size = batch_candidates.shape[0]

        # Expand input_ids for batch
        batch_input_ids = input_ids.expand(curr_batch_size, -1).clone()

        # Replace suffix tokens with candidates
        batch_input_ids[:, suffix_slice] = batch_candidates

        # Expand attention mask if provided
        batch_attention_mask = None
        if attention_mask is not None:
            batch_attention_mask = attention_mask.expand(curr_batch_size, -1)

        # Only compute logits at target positions — avoids [B, seq_len, vocab] tensor.
        seq_len = batch_input_ids.shape[1]
        num_logits_to_keep = seq_len - target_slice.start

        with torch.no_grad():
            try:
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    logits_to_keep=num_logits_to_keep,
                    use_cache=False,
                )
            except TypeError:
                try:
                    outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        num_logits_to_keep=num_logits_to_keep,
                        use_cache=False,
                    )
                except TypeError:
                    outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        use_cache=False,
                    )

            logits = outputs.logits

        # Determine target slice based on actual logits shape (same logic as
        # get_token_gradients — num_logits_to_keep may be silently ignored).
        if logits.shape[1] < seq_len:
            adj_target_slice = slice(0, target_slice.stop - target_slice.start)
        else:
            adj_target_slice = target_slice

        # Get output device from logits (may differ in multi-GPU)
        output_device = logits.device
        target_logits = logits[:, adj_target_slice, :]

        for j in range(curr_batch_size):
            candidate_logits = target_logits[j]  # [target_len, vocab_size]
            loss = F.cross_entropy(candidate_logits, target_ids.to(output_device), reduction='mean')
            all_losses.append(loss.item())

        # Free memory after each batch (critical for large models like 70B)
        del outputs, logits, target_logits, batch_input_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.tensor(all_losses, device=input_device)


def sample_candidates(
    gradients: torch.Tensor,
    embedding_matrix: torch.Tensor,
    current_suffix_ids: torch.Tensor,
    num_candidates: int,
    topk: int = 256,
    temperature: float = 1.0,
    allowed_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample candidate suffix replacements based on gradients.

    Uses the GCG algorithm:
    1. Project gradients onto token embeddings to get per-token scores
    2. Select top-k tokens per position based on negative gradient (steepest descent)
    3. Sample from top-k to create diverse candidates

    Args:
        gradients: Gradients w.r.t. suffix embeddings [suffix_len, hidden_dim]
        embedding_matrix: Token embedding matrix [vocab_size, hidden_dim]
        current_suffix_ids: Current suffix token IDs [suffix_len]
        num_candidates: Number of candidate suffixes to generate
        topk: Number of top tokens to consider per position
        temperature: Sampling temperature
        allowed_tokens: Optional mask of allowed token IDs [vocab_size]

    Returns:
        Candidate suffix token IDs [num_candidates, suffix_len]
    """
    # Ensure device consistency
    device = gradients.device
    embedding_matrix = embedding_matrix.to(device)
    current_suffix_ids = current_suffix_ids.to(device)
    if allowed_tokens is not None:
        allowed_tokens = allowed_tokens.to(device)

    suffix_len = gradients.shape[0]
    vocab_size = embedding_matrix.shape[0]

    # Project gradients onto embedding space
    # Score = -gradient @ embedding.T (negative because we want to minimize loss)
    # Higher score = token that reduces loss more
    token_scores = -torch.matmul(gradients.to(device), embedding_matrix.T)  # [suffix_len, vocab_size]

    # Apply allowed tokens mask if provided
    if allowed_tokens is not None:
        token_scores[:, ~allowed_tokens] = float('-inf')

    # Get top-k tokens per position
    topk_scores, topk_indices = torch.topk(token_scores, topk, dim=-1)  # [suffix_len, topk]

    # Generate candidates by sampling from top-k
    candidates = []
    for _ in range(num_candidates):
        candidate = current_suffix_ids.clone()

        # Randomly select positions to modify (can modify 1 or more positions)
        num_positions = torch.randint(1, min(4, suffix_len + 1), (1,)).item()
        positions = torch.randperm(suffix_len)[:num_positions]

        for pos in positions:
            # Sample from top-k tokens at this position
            if temperature > 0:
                probs = F.softmax(topk_scores[pos] / temperature, dim=-1)
                idx = torch.multinomial(probs, 1).item()
            else:
                idx = 0  # Greedy: always pick the best

            candidate[pos] = topk_indices[pos, idx]

        candidates.append(candidate)

    return torch.stack(candidates)  # [num_candidates, suffix_len]


def get_allowed_tokens(
    tokenizer,
    allow_non_ascii: bool = False,
    min_token_id: int = 3,
) -> torch.Tensor:
    """
    Get a boolean mask of allowed tokens for suffix optimization.

    Excludes:
    - Special tokens (BOS, EOS, PAD, etc.)
    - Non-ASCII tokens (optionally)
    - Control characters

    Args:
        tokenizer: The tokenizer
        allow_non_ascii: Whether to allow non-ASCII tokens
        min_token_id: Minimum token ID to allow

    Returns:
        Boolean tensor [vocab_size] where True = allowed
    """
    # Use len(tokenizer) rather than tokenizer.vocab_size.
    # For Llama-3: tokenizer.vocab_size=128000 but the model's embedding matrix
    # has 128256 rows (extra 256 special tokens added by the tokenizer).
    # Using vocab_size causes an index mismatch when masking token_scores.
    vocab_size = len(tokenizer)
    allowed = torch.ones(vocab_size, dtype=torch.bool)

    # Exclude low token IDs (usually special tokens)
    allowed[:min_token_id] = False

    # Exclude special tokens
    special_ids = []
    if tokenizer.bos_token_id is not None:
        special_ids.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        special_ids.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        special_ids.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        special_ids.append(tokenizer.unk_token_id)

    for sid in special_ids:
        if sid is not None and sid < vocab_size:
            allowed[sid] = False

    # Optionally exclude non-ASCII tokens
    if not allow_non_ascii:
        for i in range(vocab_size):
            try:
                token = tokenizer.decode([i])
                if not token.isascii():
                    allowed[i] = False
            except:
                allowed[i] = False

    return allowed


def initialize_suffix(
    tokenizer,
    num_tokens: int,
    init_string: Optional[str] = None,
    allowed_tokens: Optional[torch.Tensor] = None,
    device = "cuda",  # Can be str or torch.device
) -> torch.Tensor:
    """
    Initialize suffix tokens.

    Args:
        tokenizer: The tokenizer
        num_tokens: Number of suffix tokens
        init_string: Optional initialization string
        allowed_tokens: Optional mask of allowed tokens
        device: Device to place tensor on

    Returns:
        Initial suffix token IDs [num_tokens]
    """
    if init_string is not None:
        # Initialize from string
        suffix_ids = tokenizer.encode(init_string, add_special_tokens=False)
        suffix_ids = suffix_ids[:num_tokens]  # Truncate if too long

        # Pad if too short
        while len(suffix_ids) < num_tokens:
            suffix_ids.append(tokenizer.encode("!", add_special_tokens=False)[0])

        return torch.tensor(suffix_ids, device=device)

    else:
        # Random initialization from allowed tokens
        if allowed_tokens is not None:
            allowed_indices = torch.where(allowed_tokens)[0]
            random_indices = torch.randint(0, len(allowed_indices), (num_tokens,))
            suffix_ids = allowed_indices[random_indices]
        else:
            # Default: use common tokens
            common_tokens = "! @ # $ % ^ & * ( ) - + = [ ] { } | ; : ' \" , . < > ? / \\"
            common_ids = tokenizer.encode(common_tokens, add_special_tokens=False)
            suffix_ids = torch.tensor(
                [common_ids[i % len(common_ids)] for i in range(num_tokens)],
                device=device
            )

        return suffix_ids.to(device)


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_embedding_layer(model) -> torch.nn.Module:
    """
    Get the token embedding layer from a model.
    Handles different model architectures (Llama, Mistral, Gemma-4 VLM, etc.)
    """
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        # Llama, Mistral, Qwen style
        return model.model.embed_tokens
    elif (hasattr(model, 'model') and hasattr(model.model, 'language_model')
          and hasattr(model.model.language_model, 'embed_tokens')):
        # Gemma4ForConditionalGeneration: model.model.language_model.embed_tokens
        # (VLM nesting — text embedding lives under language_model submodule)
        return model.model.language_model.embed_tokens
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        # GPT-2 style
        return model.transformer.wte
    elif hasattr(model, 'embeddings'):
        # BERT style
        return model.embeddings.word_embeddings
    else:
        raise ValueError(f"Cannot find embedding layer for model type: {type(model)}")


def get_embedding_matrix(model) -> torch.Tensor:
    """
    Get the token embedding matrix from a model.
    """
    embedding_layer = get_embedding_layer(model)
    return embedding_layer.weight  # [vocab_size, hidden_dim]
