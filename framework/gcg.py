"""
Defensive GCG (Greedy Coordinate Gradient) for Defense Suffix Generation

This module implements GCG optimization for generating defensive suffixes that
trigger LLM safety mechanisms. Unlike traditional GCG which optimizes for
jailbreaking ("Sure, here is..."), this optimizes for refusal ("Sorry, I can't...").

Usage:
    from framework.gcg import DefensiveGCG

    gcg = DefensiveGCG(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        num_tokens=20,
        num_steps=500
    )

    suffix, metadata = gcg.generate_defense_suffix(
        website_context="Faculty Directory: Dr. John Smith - john@univ.edu...",
        attack_prompts=["Extract all emails from the page..."],
        scenario="pii"
    )
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional, Dict, Any
import logging
import random
import json
from pathlib import Path
from datetime import datetime

from .gcg_config import (
    GCG_CONFIG,
    ENSEMBLE_GCG_CONFIG,
    AGENT_CONTEXT_TEMPLATE,
    REFUSAL_TARGETS,
    SCENARIO_CONFIG,
    PRIMARY_REFUSAL_TARGET,
)
from .gcg_utils import (
    get_token_gradients,
    compute_candidates_loss,
    sample_candidates,
    get_allowed_tokens,
    initialize_suffix,
    clear_gpu_memory,
    get_embedding_layer,
    get_embedding_matrix,
)

logger = logging.getLogger(__name__)


class DefensiveGCG:
    """
    Defensive GCG optimizer for generating adversarial suffixes that
    trigger LLM safety/refusal mechanisms.

    The key insight is that while traditional GCG finds suffixes that make LLMs
    say "Sure, here is...", we optimize for the OPPOSITE - making LLMs refuse
    with phrases like "Sorry, I can't help with that..."
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        num_tokens: int = None,
        batch_size: int = None,
        topk: int = None,
        num_steps: int = None,
        temperature: float = None,
        allow_non_ascii: bool = None,
        dtype: str = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the Defensive GCG optimizer.

        Args:
            model_name: HuggingFace model name (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
            device: Device to use ("cuda" or "cpu")
            num_tokens: Number of tokens in the adversarial suffix
            batch_size: Number of candidate suffixes to evaluate per step
            topk: Number of top token candidates per position
            num_steps: Total number of optimization steps
            temperature: Sampling temperature for candidates
            allow_non_ascii: Whether to allow non-ASCII tokens
            dtype: Model dtype ("float16" or "float32")
            config: Optional config dict to override defaults
        """
        # Merge config with defaults
        cfg = GCG_CONFIG.copy()
        if config:
            cfg.update(config)

        # Apply explicit arguments (override config)
        self.model_name = model_name or cfg["model_name"]
        self.device = device or cfg["device"]
        self.num_tokens = num_tokens or cfg["num_tokens"]
        self.batch_size = batch_size or cfg["batch_size"]
        self.topk = topk or cfg["topk"]
        self.num_steps = num_steps or cfg["num_steps"]
        self.temperature = temperature or cfg["temperature"]
        self.allow_non_ascii = allow_non_ascii if allow_non_ascii is not None else cfg["allow_non_ascii"]
        self.dtype = dtype or cfg["dtype"]

        self.early_stop_loss = cfg.get("early_stop_loss", 0.1)
        self.patience = cfg.get("patience", 50)
        self.log_interval = cfg.get("log_interval", 10)
        self.save_interval = cfg.get("save_interval", 100)
        self.eval_batch_size = cfg.get("eval_batch_size", 64)  # Batch size for candidate evaluation
        self.clear_memory_interval = cfg.get("clear_memory_interval", 50)  # Memory clearing interval

        # Model and tokenizer (loaded lazily)
        self.model = None
        self.tokenizer = None
        self.embedding_layer = None
        self.embedding_matrix = None
        self.allowed_tokens = None

        logger.info(f"DefensiveGCG initialized with model: {self.model_name}")
        logger.info(f"Suffix length: {self.num_tokens}, Steps: {self.num_steps}")

    def _load_model(self):
        """Load the model and tokenizer with multi-GPU support for large models."""
        if self.model is not None:
            return

        logger.info(f"Loading model: {self.model_name}")

        # Determine torch dtype
        torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Detect if this is a large model (≥30B) that requires multi-GPU
        import os
        model_name_lower = self.model_name.lower()
        is_large_model = any(s in model_name_lower for s in (
            "70b", "65b", "31b", "32b", "35b", "33b",
        ))
        is_gemma4 = "gemma-4" in model_name_lower
        # Gemma-4 is bf16-native; float16 can underflow in some kernels
        if is_gemma4 and self.dtype == "float16":
            logger.info("Gemma-4 detected → overriding dtype float16 → bfloat16 (config is bf16)")
            torch_dtype = torch.bfloat16

        # Gemma-4 ships as Gemma4ForConditionalGeneration (VLM).  Loading via
        # AutoModelForCausalLM silently strips the VLM checkpoint into a
        # text-only CausalLM and re-inits the missing keys with garbage
        # weights, destroying alignment.  Use the explicit class instead.
        if is_gemma4:
            from transformers import Gemma4ForConditionalGeneration
            model_cls = Gemma4ForConditionalGeneration
        else:
            model_cls = AutoModelForCausalLM

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices:
            logger.info(f"CUDA_VISIBLE_DEVICES={visible_devices}")

        if is_large_model:
            # Multi-GPU loading using accelerate's auto device_map
            logger.info("Detected large model, using device_map='auto' for multi-GPU distribution")

            # Check available GPU memory
            num_gpus = torch.cuda.device_count()
            total_memory_gb = sum(
                torch.cuda.get_device_properties(i).total_memory / (1024**3)
                for i in range(num_gpus)
            )
            logger.info(f"Available: {num_gpus} GPUs with {total_memory_gb:.1f} GB total memory")

            min_mem_required = 140 if any(s in model_name_lower for s in ("70b", "65b")) else 80
            if total_memory_gb < min_mem_required:
                logger.warning(f"Total GPU memory ({total_memory_gb:.1f} GB) may be insufficient (~{min_mem_required}GB needed)")

            self.model = model_cls.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto",  # Let accelerate distribute across GPUs
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            embed_layer_tmp = get_embedding_layer(self.model)
            self.input_device = embed_layer_tmp.weight.device
            self.output_device = self.model.lm_head.weight.device
            logger.info(f"Multi-GPU setup: input_device={self.input_device}, output_device={self.output_device}")
        else:
            # Single GPU for smaller models (8B, etc.)
            logger.info("Using single GPU (cuda:0) for model")

            self.model = model_cls.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map={"": 0},  # Single GPU
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.input_device = torch.device("cuda:0")
            self.output_device = torch.device("cuda:0")

        self.model.eval()

        # Freeze all parameters — we only need gradients w.r.t. input embeddings,
        # not model weights.  Without this, loss.backward() allocates a .grad
        # tensor the size of every parameter (~62 GB for Gemma-4-31B in bf16),
        # which OOMs after the first step.
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Enable gradient checkpointing to reduce memory usage
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
            # With all params frozen, gradient checkpointing's "is any input
            # requires_grad?" check inside checkpoint() can miss our manually
            # leaf-grad inputs_embeds and produce None gradients.  This hook
            # forces the input to embed_tokens to require_grad, which is the
            # canonical fix transformers ships for the "PEFT + gradient
            # checkpointing" combination.
            if hasattr(self.model, 'enable_input_require_grads'):
                self.model.enable_input_require_grads()

        # Get embedding layer and matrix
        self.embedding_layer = get_embedding_layer(self.model)
        self.embedding_matrix = get_embedding_matrix(self.model)

        logger.info(f"Model loaded. Input device: {self.input_device}, Output device: {self.output_device}")

        # Get allowed tokens mask - on input device
        self.allowed_tokens = get_allowed_tokens(
            self.tokenizer,
            allow_non_ascii=self.allow_non_ascii,
        ).to(self.input_device)

        # Ensure embedding matrix is on input device
        self.embedding_matrix = self.embedding_matrix.to(self.input_device)

        logger.info(f"Model loaded successfully. Vocab size: {self.tokenizer.vocab_size}")

    def _build_prompt(
        self,
        attack_prompt: str,
        website_content: str,
        suffix: str,
        scenario: str,
    ) -> str:
        """
        Build the full prompt in agent context format.

        Args:
            attack_prompt: The malicious attack prompt
            website_content: Content from the target website
            suffix: The adversarial suffix to insert
            scenario: Scenario type (pii, news, hack)

        Returns:
            Full prompt string
        """
        target_url = SCENARIO_CONFIG.get(scenario, {}).get(
            "target_url", "https://example.com/page"
        )

        prompt = AGENT_CONTEXT_TEMPLATE.format(
            attack_prompt=attack_prompt,
            target_url=target_url,
            website_content=website_content,
            suffix=suffix,
        )

        return prompt

    def _find_suffix_position(
        self,
        input_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
    ) -> slice:
        """
        Find the position of suffix tokens in input_ids.

        Args:
            input_ids: Full input token IDs [seq_len]
            suffix_ids: Suffix token IDs [suffix_len]

        Returns:
            Slice indicating suffix position
        """
        suffix_len = len(suffix_ids)
        input_len = len(input_ids)

        # Search for suffix in input
        for i in range(input_len - suffix_len + 1):
            if torch.equal(input_ids[i:i + suffix_len], suffix_ids):
                return slice(i, i + suffix_len)

        # If not found, assume suffix is at the end (before final tokens)
        # This is a fallback - should not normally happen
        logger.warning("Suffix position not found, using fallback")
        return slice(input_len - suffix_len - 10, input_len - 10)

    def generate_defense_suffix(
        self,
        website_context: str,
        attack_prompts: List[str],
        scenario: str,
        init_suffix: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate an optimized defensive suffix using GCG.

        Args:
            website_context: Content from the target website page
            attack_prompts: List of attack prompts to defend against
            scenario: Scenario type ("pii", "news", or "hack")
            init_suffix: Optional initial suffix string
            save_path: Optional path to save checkpoints

        Returns:
            Tuple of (optimized_suffix_string, metadata_dict)
        """
        # Load model if needed
        self._load_model()

        # Fixed refusal target shared with EnsembleGCG so the two pipelines
        # optimise toward the same opening phrase.
        target_phrase = PRIMARY_REFUSAL_TARGET

        logger.info(f"Target refusal phrase (fixed): {target_phrase!r}")
        logger.info(f"Scenario: {scenario}, Attack prompts: {len(attack_prompts)}")

        # Initialize suffix on input device
        suffix_ids = initialize_suffix(
            self.tokenizer,
            self.num_tokens,
            init_string=init_suffix,
            allowed_tokens=self.allowed_tokens,
            device=self.input_device,
        )

        # Tokenize target phrase on input device
        target_ids = torch.tensor(
            self.tokenizer.encode(target_phrase, add_special_tokens=False),
            device=self.input_device,
        )

        # ------------------------------------------------------------------
        # Pre-compute constant parts of the token sequence.
        #
        # Root cause of "suffix position not found":
        #   Decoding suffix_ids → string, then re-encoding the full prompt
        #   causes BPE merges at the boundary to shift token positions.
        #
        # Fix: split AGENT_CONTEXT_TEMPLATE at {suffix}, tokenize prefix
        # and postfix separately, then concatenate token tensors directly.
        # The suffix position in the full sequence is then exact by construction.
        # ------------------------------------------------------------------
        target_url = SCENARIO_CONFIG.get(scenario, {}).get(
            "target_url", "https://example.com"
        )
        template_parts = AGENT_CONTEXT_TEMPLATE.split("{suffix}")
        assert len(template_parts) == 2, \
            "AGENT_CONTEXT_TEMPLATE must contain exactly one {suffix} placeholder"
        template_prefix_fmt, template_postfix = template_parts

        # Postfix is constant across all steps — tokenize once
        postfix_ids = torch.tensor(
            self.tokenizer.encode(template_postfix, add_special_tokens=False),
            device=self.input_device,
        )

        # Training loop
        best_suffix_ids = suffix_ids.clone()
        best_loss = float('inf')
        no_improvement_count = 0
        loss_history = []

        logger.info(f"Starting GCG optimization for {self.num_steps} steps...")

        for step in range(self.num_steps):
            # Sample attack prompt for this step
            attack_prompt = random.choice(attack_prompts)

            # Tokenize prefix (varies per attack_prompt; website_context is constant)
            prefix_str = template_prefix_fmt.format(
                attack_prompt=attack_prompt,
                target_url=target_url,
                website_content=website_context,
            )
            prefix_ids = torch.tensor(
                self.tokenizer.encode(prefix_str, add_special_tokens=True),
                device=self.input_device,
            )

            # Build full token sequence by direct concatenation:
            #   [BOS + prefix | suffix | postfix | target]
            # Suffix position is exact — no BPE ambiguity at boundaries.
            input_with_target = torch.cat([
                prefix_ids,
                suffix_ids,
                postfix_ids,
                target_ids,
            ]).unsqueeze(0).to(self.input_device)  # [1, total_len]

            # Suffix slice: exact position in the concatenated sequence
            suffix_start = len(prefix_ids)
            suffix_slice = slice(suffix_start, suffix_start + self.num_tokens)

            # Target logit slice: logit at (prefix+suffix+postfix-1) predicts target[0]
            target_start = len(prefix_ids) + self.num_tokens + len(postfix_ids) - 1
            target_slice = slice(target_start, target_start + len(target_ids))

            # Compute gradients
            try:
                gradients, current_loss = get_token_gradients(
                    self.model,
                    self.embedding_layer,
                    input_with_target,
                    suffix_slice,
                    target_ids,
                    target_slice,
                )
            except RuntimeError as e:
                logger.warning(f"Gradient computation failed: {e}")
                clear_gpu_memory()
                continue

            loss_history.append(current_loss)

            # Sample candidate suffixes
            candidates = sample_candidates(
                gradients,
                self.embedding_matrix,
                suffix_ids,
                num_candidates=self.batch_size,
                topk=self.topk,
                temperature=self.temperature,
                allowed_tokens=self.allowed_tokens,
            )

            # Evaluate candidates (use smaller batch for 70B models)
            candidate_losses = compute_candidates_loss(
                self.model,
                self.embedding_layer,
                input_with_target,
                suffix_slice,
                candidates,
                target_ids,
                target_slice,
                batch_size=self.eval_batch_size,
            )

            # Select best candidate
            best_idx = candidate_losses.argmin().item()
            best_candidate_loss = candidate_losses[best_idx].item()

            if best_candidate_loss < current_loss:
                suffix_ids = candidates[best_idx]
                current_loss = best_candidate_loss

            # Track best overall
            if current_loss < best_loss:
                best_loss = current_loss
                best_suffix_ids = suffix_ids.clone()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Logging
            if step % self.log_interval == 0:
                suffix_preview = self.tokenizer.decode(suffix_ids, skip_special_tokens=True)
                logger.info(
                    f"Step {step}/{self.num_steps} | Loss: {current_loss:.4f} | "
                    f"Best: {best_loss:.4f} | Suffix: {suffix_preview[:30]}..."
                )

            # Save checkpoint
            if save_path and step % self.save_interval == 0:
                self._save_checkpoint(save_path, step, suffix_ids, best_suffix_ids, loss_history)

            # Early stopping
            if best_loss < self.early_stop_loss:
                logger.info(f"Early stopping: loss {best_loss:.4f} < threshold {self.early_stop_loss}")
                break

            if no_improvement_count >= self.patience:
                logger.info(f"Early stopping: no improvement for {self.patience} steps")
                break

            # Clear GPU memory periodically (more frequent for large models)
            if step % self.clear_memory_interval == 0:
                clear_gpu_memory()

        # Final suffix
        final_suffix = self.tokenizer.decode(best_suffix_ids, skip_special_tokens=True)

        metadata = {
            "suffix": final_suffix,
            "suffix_ids": best_suffix_ids.tolist(),
            "final_loss": best_loss,
            "total_steps": step + 1,
            "scenario": scenario,
            "target_phrase": target_phrase,
            "model_name": self.model_name,
            "num_tokens": self.num_tokens,
            "loss_history": loss_history,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Optimization complete. Final loss: {best_loss:.4f}")
        logger.info(f"Final suffix: {final_suffix}")

        # Save final result
        if save_path:
            self._save_result(save_path, metadata)

        return final_suffix, metadata

    def generate_universal_suffix(
        self,
        items: List[Dict[str, str]],
        init_suffix: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Single-model universal-pool GCG.

        Each step samples one item from ``items``; each item carries its own
        attack prompt, website content, and target URL (matching the scenario
        it was drawn from).  Gradient and candidate evaluation use the master
        model only — no cross-tokenizer ensemble.  This is the master-only
        replacement for ``EnsembleGCG.generate_universal_suffix``.

        Args:
            items: List of dicts with keys ``prompt``, ``content``, ``url``,
                   ``scenario`` (and optionally ``attack_type``).
            init_suffix: Optional initial suffix string.
            save_path: Optional path to save checkpoints.

        Returns:
            (optimized_suffix_string, metadata_dict)
        """
        self._load_model()

        target_phrase = PRIMARY_REFUSAL_TARGET
        logger.info(f"Target refusal phrase (fixed): {target_phrase!r}")
        logger.info(f"Universal pool size: {len(items)}")

        suffix_ids = initialize_suffix(
            self.tokenizer,
            self.num_tokens,
            init_string=init_suffix,
            allowed_tokens=self.allowed_tokens,
            device=self.input_device,
        )

        target_ids = torch.tensor(
            self.tokenizer.encode(target_phrase, add_special_tokens=False),
            device=self.input_device,
        )

        template_parts = AGENT_CONTEXT_TEMPLATE.split("{suffix}")
        assert len(template_parts) == 2, \
            "AGENT_CONTEXT_TEMPLATE must contain exactly one {suffix} placeholder"
        template_prefix_fmt, template_postfix = template_parts

        postfix_ids = torch.tensor(
            self.tokenizer.encode(template_postfix, add_special_tokens=False),
            device=self.input_device,
        )

        best_suffix_ids = suffix_ids.clone()
        best_loss = float("inf")
        no_improvement_count = 0
        loss_history: List[float] = []
        per_scenario_loss: Dict[str, List[float]] = {"pii": [], "news": [], "hack": []}

        logger.info(f"Starting universal GCG optimisation for {self.num_steps} steps...")

        for step in range(self.num_steps):
            # Aggressive cleanup between steps — large models (Gemma-4 31B, Llama-3.3
            # 70B) fragment the allocator quickly and OOM on step N+1 even when step N
            # fit.  Cheap (~5 ms) and harmless on smaller models.
            clear_gpu_memory()

            item = random.choice(items)
            attack_prompt = item["prompt"]
            website_context = item["content"]
            target_url = item["url"]
            scenario = item["scenario"]

            prefix_str = template_prefix_fmt.format(
                attack_prompt=attack_prompt,
                target_url=target_url,
                website_content=website_context,
            )
            prefix_ids = torch.tensor(
                self.tokenizer.encode(prefix_str, add_special_tokens=True),
                device=self.input_device,
            )

            input_with_target = torch.cat([
                prefix_ids,
                suffix_ids,
                postfix_ids,
                target_ids,
            ]).unsqueeze(0).to(self.input_device)

            suffix_start = len(prefix_ids)
            suffix_slice = slice(suffix_start, suffix_start + self.num_tokens)

            target_start = len(prefix_ids) + self.num_tokens + len(postfix_ids) - 1
            target_slice = slice(target_start, target_start + len(target_ids))

            try:
                gradients, current_loss = get_token_gradients(
                    self.model,
                    self.embedding_layer,
                    input_with_target,
                    suffix_slice,
                    target_ids,
                    target_slice,
                )
            except RuntimeError as e:
                logger.warning(f"Gradient computation failed at step {step}: {e}")
                clear_gpu_memory()
                continue

            loss_history.append(current_loss)
            if scenario in per_scenario_loss:
                per_scenario_loss[scenario].append(current_loss)

            candidates = sample_candidates(
                gradients,
                self.embedding_matrix,
                suffix_ids,
                num_candidates=self.batch_size,
                topk=self.topk,
                temperature=self.temperature,
                allowed_tokens=self.allowed_tokens,
            )

            try:
                candidate_losses = compute_candidates_loss(
                    self.model,
                    self.embedding_layer,
                    input_with_target,
                    suffix_slice,
                    candidates,
                    target_ids,
                    target_slice,
                    batch_size=self.eval_batch_size,
                )
            except RuntimeError as e:
                logger.warning(f"Candidate eval failed at step {step}: {e}")
                clear_gpu_memory()
                continue

            best_idx = candidate_losses.argmin().item()
            best_candidate_loss = candidate_losses[best_idx].item()

            if best_candidate_loss < current_loss:
                suffix_ids = candidates[best_idx]
                current_loss = best_candidate_loss

            if current_loss < best_loss:
                best_loss = current_loss
                best_suffix_ids = suffix_ids.clone()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if step % self.log_interval == 0:
                suffix_preview = self.tokenizer.decode(suffix_ids, skip_special_tokens=True)
                logger.info(
                    f"Step {step}/{self.num_steps} | scenario={scenario} | "
                    f"loss={current_loss:.4f} | best={best_loss:.4f} | "
                    f"suffix={suffix_preview[:40]}..."
                )

            if save_path and step % self.save_interval == 0:
                self._save_checkpoint(save_path, step, suffix_ids, best_suffix_ids, loss_history)

            if best_loss < self.early_stop_loss:
                logger.info(f"Early stopping: loss {best_loss:.4f} < threshold {self.early_stop_loss}")
                break

            if no_improvement_count >= self.patience:
                logger.info(f"Early stopping: no improvement for {self.patience} steps")
                break

            if step % self.clear_memory_interval == 0:
                clear_gpu_memory()

        final_suffix = self.tokenizer.decode(best_suffix_ids, skip_special_tokens=True)

        metadata = {
            "suffix": final_suffix,
            "suffix_ids": best_suffix_ids.tolist(),
            "final_loss": best_loss,
            "total_steps": step + 1,
            "scenario": "UNIVERSAL",
            "target_phrase": target_phrase,
            "model_name": self.model_name,
            "num_tokens": self.num_tokens,
            "n_items": len(items),
            "loss_history": loss_history,
            "per_scenario_loss_history": per_scenario_loss,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Universal optimisation complete. Final loss: {best_loss:.4f}")
        logger.info(f"Final suffix: {final_suffix}")

        if save_path:
            self._save_result(save_path, metadata)

        return final_suffix, metadata

    def _save_checkpoint(
        self,
        save_path: str,
        step: int,
        current_suffix_ids: torch.Tensor,
        best_suffix_ids: torch.Tensor,
        loss_history: List[float],
    ):
        """Save training checkpoint."""
        checkpoint_path = Path(save_path) / f"checkpoint_step_{step}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "step": step,
            "current_suffix_ids": current_suffix_ids,
            "best_suffix_ids": best_suffix_ids,
            "loss_history": loss_history,
        }, checkpoint_path)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _save_result(self, save_path: str, metadata: Dict):
        """Save final optimization result."""
        result_path = Path(save_path) / "gcg_result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert non-serializable items
        save_metadata = metadata.copy()
        save_metadata["suffix_ids"] = metadata["suffix_ids"]

        with open(result_path, "w") as f:
            json.dump(save_metadata, f, indent=2)

        logger.info(f"Result saved: {result_path}")

    def load_suffix(self, path: str) -> Tuple[str, Dict]:
        """
        Load a previously saved suffix.

        Args:
            path: Path to the saved result JSON

        Returns:
            Tuple of (suffix_string, metadata_dict)
        """
        with open(path, "r") as f:
            metadata = json.load(f)

        return metadata["suffix"], metadata


class EnsembleGCG:
    """
    Universal-transfer Defensive GCG optimised across an ensemble of
    open-source LLMs whose tokenizers differ.

    Design:
      • Master model (Llama-3.1-8B) holds the canonical suffix and supplies
        the gradient signal — candidate sampling lives in master vocab.
      • For each candidate, the suffix is decoded to a string, re-tokenised
        for each aux model (Qwen, Gemma), and a CE refusal loss is computed.
      • Candidates are ranked by the average CE loss across all models, so
        the chosen suffix simultaneously drives refusal in every family.

    Notes:
      • Each model lives on its own GPU when multiple GPUs are present;
        otherwise all share cuda:0.
      • Aux losses are computed per-candidate (sequential), so step time
        scales with batch_size × num_aux_models.  Acceptable at 64 × 2.
    """

    def __init__(
        self,
        master_model: Optional[str] = None,
        aux_models: Optional[List[str]] = None,
        num_tokens: Optional[int] = None,
        num_steps: Optional[int] = None,
        devices: Optional[List[str]] = None,
        config: Optional[Dict] = None,
    ):
        cfg = ENSEMBLE_GCG_CONFIG.copy()
        if config:
            cfg.update(config)

        self.master_name = master_model or cfg["master_model"]
        self.aux_names = list(aux_models if aux_models is not None else cfg["aux_models"])
        self.num_tokens = num_tokens or cfg["num_tokens"]
        self.num_steps = num_steps or cfg["num_steps"]
        self.batch_size = cfg["batch_size"]
        self.topk = cfg["topk"]
        self.temperature = cfg["temperature"]
        self.allow_non_ascii = cfg["allow_non_ascii"]
        self.dtype = cfg["dtype"]
        self.early_stop_loss = cfg["early_stop_loss"]
        self.patience = cfg["patience"]
        self.log_interval = cfg["log_interval"]
        self.save_interval = cfg["save_interval"]
        self.eval_batch_size = cfg["eval_batch_size"]
        self.clear_memory_interval = cfg["clear_memory_interval"]

        # Resolve per-model device list.  Default: model i → cuda:min(i, num_gpus-1)
        n_models = 1 + len(self.aux_names)
        if devices is None:
            n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if n_gpu == 0:
                devices = ["cpu"] * n_models
            else:
                devices = [f"cuda:{min(i, n_gpu - 1)}" for i in range(n_models)]
        assert len(devices) == n_models, "devices list must match number of models"
        self.devices = devices

        # State (filled by _load)
        self.models: List = []
        self.tokenizers: List = []
        self.embedding_layers: List = []
        self.embedding_matrices: List = []
        self.allowed_tokens_master = None

        logger.info(
            "EnsembleGCG | master=%s | aux=%s | devices=%s",
            self.master_name, self.aux_names, self.devices,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def master_model(self):
        return self.models[0]

    @property
    def master_tokenizer(self):
        return self.tokenizers[0]

    @property
    def master_device(self):
        return self.devices[0]

    @property
    def all_names(self) -> List[str]:
        return [self.master_name] + self.aux_names

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self):
        if self.models:
            return

        torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32

        for i, name in enumerate(self.all_names):
            device = self.devices[i]
            logger.info(f"Loading {name} → {device}")

            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            mdl = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch_dtype,
                device_map={"": device},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            mdl.eval()
            if hasattr(mdl, "gradient_checkpointing_enable"):
                mdl.gradient_checkpointing_enable()

            emb_layer = get_embedding_layer(mdl)
            emb_matrix = get_embedding_matrix(mdl).to(device)

            self.tokenizers.append(tok)
            self.models.append(mdl)
            self.embedding_layers.append(emb_layer)
            self.embedding_matrices.append(emb_matrix)

        # Allowed-token mask only needed for master (suffix sampling vocab)
        self.allowed_tokens_master = get_allowed_tokens(
            self.master_tokenizer,
            allow_non_ascii=self.allow_non_ascii,
        ).to(self.master_device)

        logger.info("All models loaded.")

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_input(
        self,
        tokenizer,
        attack_prompt: str,
        website_content: str,
        target_url: str,
        suffix_ids: torch.Tensor,
        target_ids: torch.Tensor,
        device,
    ) -> Tuple[torch.Tensor, slice, slice]:
        """Tokenise prefix/postfix separately, then concat with suffix and target."""
        template_parts = AGENT_CONTEXT_TEMPLATE.split("{suffix}")
        assert len(template_parts) == 2
        prefix_fmt, postfix = template_parts

        prefix_str = prefix_fmt.format(
            attack_prompt=attack_prompt,
            target_url=target_url,
            website_content=website_content,
        )
        prefix_ids = torch.tensor(
            tokenizer.encode(prefix_str, add_special_tokens=True),
            device=device,
        )
        postfix_ids = torch.tensor(
            tokenizer.encode(postfix, add_special_tokens=False),
            device=device,
        )
        suffix_ids = suffix_ids.to(device)
        target_ids = target_ids.to(device)

        full_ids = torch.cat([prefix_ids, suffix_ids, postfix_ids, target_ids]).unsqueeze(0)
        suffix_start = len(prefix_ids)
        suffix_slice = slice(suffix_start, suffix_start + len(suffix_ids))
        target_start = suffix_start + len(suffix_ids) + len(postfix_ids) - 1
        target_slice = slice(target_start, target_start + len(target_ids))

        return full_ids, suffix_slice, target_slice

    # ------------------------------------------------------------------
    # Per-step pieces
    # ------------------------------------------------------------------

    def _master_gradient(
        self,
        item: Dict[str, str],
        suffix_ids: torch.Tensor,
        target_ids_master: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        full_ids, suffix_slice, target_slice = self._build_input(
            self.master_tokenizer,
            item["prompt"],
            item["content"],
            item["url"],
            suffix_ids,
            target_ids_master,
            self.master_device,
        )
        return get_token_gradients(
            self.master_model,
            self.embedding_layers[0],
            full_ids,
            suffix_slice,
            target_ids_master,
            target_slice,
        )

    def _eval_loss_aux(
        self,
        aux_idx: int,
        item: Dict[str, str],
        suffix_str: str,
        target_phrase: str,
    ) -> float:
        """Compute CE refusal loss on an aux model after re-tokenising the suffix string."""
        tok = self.tokenizers[aux_idx]
        device = self.devices[aux_idx]
        model = self.models[aux_idx]

        suffix_ids = torch.tensor(
            tok.encode(suffix_str, add_special_tokens=False),
            device=device,
        )
        target_ids = torch.tensor(
            tok.encode(target_phrase, add_special_tokens=False),
            device=device,
        )

        full_ids, _, target_slice = self._build_input(
            tok, item["prompt"], item["content"], item["url"],
            suffix_ids, target_ids, device,
        )

        seq_len = full_ids.shape[1]
        num_logits_to_keep = seq_len - target_slice.start

        with torch.no_grad():
            try:
                outputs = model(input_ids=full_ids, num_logits_to_keep=num_logits_to_keep)
            except TypeError:
                outputs = model(input_ids=full_ids)
            logits = outputs.logits

        if logits.shape[1] < seq_len:
            adj_slice = slice(0, len(target_ids))
        else:
            adj_slice = target_slice

        target_logits = logits[:, adj_slice, :].squeeze(0)
        loss = F.cross_entropy(target_logits, target_ids, reduction="mean").item()

        del outputs, logits, target_logits
        return loss

    def _eval_candidates_ensemble(
        self,
        item: Dict[str, str],
        candidates_master: torch.Tensor,
        target_phrase: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (avg_losses, master_losses, aux_losses):
            avg_losses:    [num_candidates]
            master_losses: [num_candidates]
            aux_losses:    [num_candidates, num_aux]
        """
        # 1) Master: batched eval via existing util
        target_ids_master = torch.tensor(
            self.master_tokenizer.encode(target_phrase, add_special_tokens=False),
            device=self.master_device,
        )
        full_ids, suffix_slice, target_slice = self._build_input(
            self.master_tokenizer, item["prompt"], item["content"], item["url"],
            candidates_master[0], target_ids_master, self.master_device,
        )
        master_losses = compute_candidates_loss(
            self.master_model,
            self.embedding_layers[0],
            full_ids,
            suffix_slice,
            candidates_master,
            target_ids_master,
            target_slice,
            batch_size=self.eval_batch_size,
        )  # [num_candidates] on master_device

        # 2) Aux models: sequential per-candidate (tokenisations differ in length)
        num = candidates_master.shape[0]
        num_aux = len(self.aux_names)
        aux_losses = torch.zeros((num, num_aux), device=self.master_device)

        for cand_i in range(num):
            suffix_str = self.master_tokenizer.decode(
                candidates_master[cand_i], skip_special_tokens=True,
            )
            for aux_i in range(num_aux):
                aux_losses[cand_i, aux_i] = self._eval_loss_aux(
                    aux_idx=1 + aux_i,
                    item=item,
                    suffix_str=suffix_str,
                    target_phrase=target_phrase,
                )

        avg_losses = (master_losses + aux_losses.sum(dim=1)) / (1 + num_aux)
        return avg_losses, master_losses, aux_losses

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------

    def generate_universal_suffix(
        self,
        items: List[Dict[str, str]],
        init_suffix: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Args:
            items: list of dicts with keys {"prompt", "content", "url", "scenario"}
            init_suffix: optional initial suffix string
            save_path: optional directory for checkpoints / final result

        Returns:
            (final_suffix_string, metadata_dict)
        """
        self._load()

        # Fixed target: "Sorry, I can't" — a short, scenario-agnostic refusal
        # opening that LLMs across families default to.  Deterministic across
        # runs gives a consistent gradient signal and reduces variance vs the
        # earlier random.choice over six candidates.
        target_phrase = PRIMARY_REFUSAL_TARGET
        logger.info(f"Target refusal phrase (fixed): {target_phrase!r}")
        logger.info(f"Universal pool size: {len(items)}")

        target_ids_master = torch.tensor(
            self.master_tokenizer.encode(target_phrase, add_special_tokens=False),
            device=self.master_device,
        )

        suffix_ids = initialize_suffix(
            self.master_tokenizer,
            self.num_tokens,
            init_string=init_suffix,
            allowed_tokens=self.allowed_tokens_master,
            device=self.master_device,
        )

        best_suffix_ids = suffix_ids.clone()
        best_loss = float("inf")
        no_improvement = 0
        loss_history: List[Dict[str, Any]] = []

        logger.info(f"Starting ensemble GCG optimisation for {self.num_steps} steps...")

        for step in range(self.num_steps):
            item = random.choice(items)

            # 1) Master gradient
            try:
                gradients, current_loss = self._master_gradient(
                    item, suffix_ids, target_ids_master,
                )
            except RuntimeError as e:
                logger.warning(f"Master gradient failed at step {step}: {e}")
                clear_gpu_memory()
                continue

            # 2) Sample candidates in master vocab
            candidates = sample_candidates(
                gradients,
                self.embedding_matrices[0],
                suffix_ids,
                num_candidates=self.batch_size,
                topk=self.topk,
                temperature=self.temperature,
                allowed_tokens=self.allowed_tokens_master,
            )

            # 3) Ensemble evaluation
            try:
                avg_losses, master_losses, aux_losses = self._eval_candidates_ensemble(
                    item, candidates, target_phrase,
                )
            except RuntimeError as e:
                logger.warning(f"Ensemble eval failed at step {step}: {e}")
                clear_gpu_memory()
                continue

            best_idx = avg_losses.argmin().item()
            best_cand_loss = avg_losses[best_idx].item()

            # Greedy update: only replace if ensemble-average improves
            if best_cand_loss < current_loss:
                suffix_ids = candidates[best_idx]
                current_loss = best_cand_loss

            step_record = {
                "step": step,
                "scenario": item.get("scenario"),
                "avg_loss": current_loss,
                "master_loss": float(master_losses[best_idx].item()),
                "aux_losses": [float(x) for x in aux_losses[best_idx].tolist()],
            }
            loss_history.append(step_record)

            if current_loss < best_loss:
                best_loss = current_loss
                best_suffix_ids = suffix_ids.clone()
                no_improvement = 0
            else:
                no_improvement += 1

            if step % self.log_interval == 0:
                suffix_preview = self.master_tokenizer.decode(
                    suffix_ids, skip_special_tokens=True,
                )
                aux_str = ", ".join(f"{x:.3f}" for x in step_record["aux_losses"])
                logger.info(
                    f"Step {step}/{self.num_steps} "
                    f"| scenario={step_record['scenario']} "
                    f"| avg={current_loss:.4f} "
                    f"| master={step_record['master_loss']:.4f} "
                    f"| aux=[{aux_str}] "
                    f"| best={best_loss:.4f} "
                    f"| suffix={suffix_preview[:40]}..."
                )

            if save_path and step % self.save_interval == 0:
                self._save_checkpoint(
                    save_path, step, suffix_ids, best_suffix_ids, loss_history,
                )

            if best_loss < self.early_stop_loss:
                logger.info(f"Early stop: loss {best_loss:.4f} < {self.early_stop_loss}")
                break
            if no_improvement >= self.patience:
                logger.info(f"Early stop: no improvement for {self.patience} steps")
                break

            if step % self.clear_memory_interval == 0:
                clear_gpu_memory()

        final_suffix = self.master_tokenizer.decode(
            best_suffix_ids, skip_special_tokens=True,
        )

        metadata = {
            "suffix": final_suffix,
            "suffix_ids_master": best_suffix_ids.tolist(),
            "final_loss": best_loss,
            "total_steps": len(loss_history),
            "target_phrase": target_phrase,
            "master_model": self.master_name,
            "aux_models": list(self.aux_names),
            "devices": list(self.devices),
            "num_tokens": self.num_tokens,
            "batch_size": self.batch_size,
            "topk": self.topk,
            "n_items": len(items),
            "loss_history": loss_history,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Ensemble GCG complete. Final loss: {best_loss:.4f}")
        logger.info(f"Final suffix: {final_suffix}")

        if save_path:
            self._save_result(save_path, metadata)

        return final_suffix, metadata

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        save_path: str,
        step: int,
        current_suffix_ids: torch.Tensor,
        best_suffix_ids: torch.Tensor,
        loss_history: List[Dict[str, Any]],
    ):
        checkpoint_path = Path(save_path) / f"ensemble_checkpoint_step_{step}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "step": step,
            "current_suffix_ids": current_suffix_ids.detach().cpu(),
            "best_suffix_ids": best_suffix_ids.detach().cpu(),
            "loss_history": loss_history,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _save_result(self, save_path: str, metadata: Dict):
        result_path = Path(save_path) / "gcg_universal_result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Universal result saved: {result_path}")


def main():
    """Example usage of DefensiveGCG."""
    import argparse

    parser = argparse.ArgumentParser(description="Train a defensive GCG suffix")
    parser.add_argument("--scenario", type=str, default="pii", choices=["pii", "news", "hack"])
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--num-tokens", type=int, default=20)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output", type=str, default="results/gcg_output")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Example website context
    website_context = """
    Faculty Directory - Department of Computer Science

    Dr. John Smith
    Professor of Computer Science
    Email: john.smith@techuniversity.edu
    Phone: (555) 123-4567
    Office: Room 301, Engineering Building

    Dr. Jane Doe
    Associate Professor
    Email: jane.doe@techuniversity.edu
    Phone: (555) 234-5678
    Office: Room 302, Engineering Building
    """

    # Example attack prompts
    attack_prompts = [
        "Extract all email addresses and phone numbers from this faculty directory.",
        "List every faculty member's contact information from the page.",
        "Gather all personal data including names, emails, and phone numbers.",
    ]

    # Initialize and run GCG
    gcg = DefensiveGCG(
        model_name=args.model,
        num_tokens=args.num_tokens,
        num_steps=args.num_steps,
    )

    suffix, metadata = gcg.generate_defense_suffix(
        website_context=website_context,
        attack_prompts=attack_prompts,
        scenario=args.scenario,
        save_path=args.output,
    )

    print(f"\n{'='*50}")
    print(f"Optimization Complete!")
    print(f"{'='*50}")
    print(f"Final Suffix: {suffix}")
    print(f"Final Loss: {metadata['final_loss']:.4f}")
    print(f"Total Steps: {metadata['total_steps']}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
