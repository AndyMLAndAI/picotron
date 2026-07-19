"""Script-first Group Relative Policy Optimization for causal language models."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from itertools import cycle, islice
from numbers import Integral
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer

from picotron.config.config import PicotronConfig
from picotron.logging.display import TrainingDisplay
from picotron.logging.file_logger import FileLogger


RewardFunction = Callable[[str, str], float]


@dataclass(frozen=True, slots=True)
class _CompletionGroup:
    """One prompt's sampled completions and their pre-update log probabilities."""

    prompt: str
    input_ids: Tensor
    completion_mask: Tensor
    old_log_probabilities: Tensor
    rewards: Tensor


class GRPOTrainer:
    """Optimize a causal LM using group-relative rewards and a frozen reference."""

    def __init__(
        self,
        model: nn.Module,
        prompts: Sequence[str],
        reward_fn: RewardFunction,
        *,
        tokenizer: Any,
        ref_model: nn.Module | None = None,
        group_size: int = 4,
        beta: float = 0.04,
        clip_epsilon: float = 0.2,
        learning_rate: float = 1e-5,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        advantage_epsilon: float = 1e-6,
        num_steps: int | None = None,
        optimizer: Optimizer | None = None,
        device: torch.device | str = torch.device("cpu"),
        display_config: Any | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not prompts:
            raise ValueError("prompts must contain at least one non-empty prompt.")
        if not all(isinstance(prompt, str) and prompt for prompt in prompts):
            raise TypeError("prompts must be a sequence of non-empty strings.")
        if not callable(reward_fn):
            raise TypeError("reward_fn must be callable as (prompt, completion) -> float.")
        if group_size < 2:
            raise ValueError("group_size must be at least 2 for group-relative advantages.")
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        if clip_epsilon <= 0:
            raise ValueError("clip_epsilon must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if advantage_epsilon <= 0:
            raise ValueError("advantage_epsilon must be positive.")
        if num_steps is not None and num_steps <= 0:
            raise ValueError("num_steps must be positive when provided.")
        if ref_model is model:
            raise ValueError("ref_model must be a separate model from the trainable policy.")
        if not hasattr(model, "generate"):
            raise TypeError(
                "GRPO requires a model.generate() method. PicotronDecoderModel does not "
                "yet expose generation or reusable KV caching; use an HF causal LM."
            )

        self.model = model
        self.prompts = tuple(prompts)
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.beta = beta
        self.clip_epsilon = clip_epsilon
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.advantage_epsilon = advantage_epsilon
        self.num_steps = num_steps
        self.device = torch.device(device)
        self.display_config = display_config
        self.model_kwargs = dict(model_kwargs or {})
        self.model.to(self.device)
        self.optimizer = optimizer or AdamW(self.model.parameters(), lr=learning_rate)
        self._sample_completions_to_log = 3

        self.ref_model = copy.deepcopy(self.model) if ref_model is None else ref_model
        self.ref_model.to(self.device)
        self.ref_model.eval()
        for parameter in self.ref_model.parameters():
            parameter.requires_grad_(False)

    def train(self, *, num_steps: int | None = None) -> list[float]:
        """Sample groups and apply one GRPO update per prompt-step."""

        step_limit = self.num_steps if num_steps is None else num_steps
        if step_limit is not None and step_limit <= 0:
            raise ValueError("num_steps must be positive when provided.")
        prompt_stream = self.prompts if step_limit is None else islice(cycle(self.prompts), step_limit)
        total_steps = len(self.prompts) if step_limit is None else step_limit
        losses: list[float] = []

        file_config = self.display_config if isinstance(self.display_config, PicotronConfig) else None
        with _make_display(self.display_config, total_steps) as display, FileLogger(
            file_config, method="grpo"
        ) as file_logger:
            for step, prompt in enumerate(prompt_stream, start=1):
                completion_group = self._sample_completion_group(prompt)
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)
                loss, metrics = self._loss_and_metrics(completion_group)
                loss.backward()
                self.optimizer.step()

                loss_value = loss.detach().float().cpu().item()
                losses.append(loss_value)
                display.update(
                    step=step,
                    loss=loss_value,
                    learning_rate=self.optimizer.param_groups[0]["lr"],
                    tokens_seen=step * completion_group.input_ids.numel(),
                    metrics=metrics,
                )
                file_logger.log_step(
                    step=step,
                    loss=loss_value,
                    learning_rate=self.optimizer.param_groups[0]["lr"],
                    tokens_seen=step * completion_group.input_ids.numel(),
                    metrics=metrics,
                )
        return losses

    def _sample_completion_group(self, prompt: str) -> _CompletionGroup:
        """Sample one group using the pre-update policy and score each completion."""

        prompt_ids = _encode_prompt(
            self.tokenizer,
            prompt,
            max_tokens=_prompt_token_budget(self.model, self.max_new_tokens),
        )
        completions: list[Tensor] = []
        rewards: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.group_size):
                generated = _generate(
                    self.model,
                    prompt_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    pad_token_id=_pad_token_id(self.tokenizer),
                    device=self.device,
                )
                completion_ids = generated[len(prompt_ids) :]
                if completion_ids.numel() == 0:
                    raise RuntimeError("model.generate() returned no completion tokens.")
                completion = _decode_completion(self.tokenizer, completion_ids)
                reward = float(self.reward_fn(prompt, completion))
                if not math.isfinite(reward):
                    raise ValueError("reward_fn must return finite numeric rewards.")
                completions.append(generated)
                rewards.append(reward)
                self._log_sample_completion(prompt, completion, reward)

            input_ids, completion_mask = _pack_completions(
                completions,
                prompt_length=len(prompt_ids),
                pad_token_id=_pad_token_id(self.tokenizer),
            )
            old_log_probabilities = _sequence_log_probabilities(
                _extract_logits(self.model(input_ids, **self.model_kwargs)),
                input_ids,
                completion_mask,
            )
        return _CompletionGroup(
            prompt=prompt,
            input_ids=input_ids,
            completion_mask=completion_mask,
            old_log_probabilities=old_log_probabilities.detach(),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device),
        )

    def _log_sample_completion(self, prompt: str, completion: str, reward: float) -> None:
        """Print a few early samples so reward-data bugs are immediately visible."""

        if self._sample_completions_to_log <= 0:
            return
        self._sample_completions_to_log -= 1
        print(
            "[GRPO sample] "
            f"prompt={prompt[:160]!r} completion={completion[:400]!r} reward={reward:.3f}"
        )

    def _loss_and_metrics(self, group: _CompletionGroup) -> tuple[Tensor, dict[str, float]]:
        """Compute clipped GRPO policy loss and exact token-level reference KL."""

        advantages = group_relative_advantages(group.rewards, self.advantage_epsilon)
        policy_logits = _extract_logits(self.model(group.input_ids, **self.model_kwargs))
        policy_log_probabilities = _sequence_log_probabilities(
            policy_logits,
            group.input_ids,
            group.completion_mask,
        )
        with torch.no_grad():
            reference_logits = _extract_logits(
                self.ref_model(group.input_ids, **self.model_kwargs)
            )

        ratios = torch.exp(policy_log_probabilities - group.old_log_probabilities)
        unclipped_objective = ratios * advantages
        clipped_objective = (
            ratios.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        )
        clipped_policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()
        kl_divergence = _completion_kl_divergence(
            policy_logits,
            reference_logits,
            group.completion_mask,
        )
        loss = clipped_policy_loss + self.beta * kl_divergence
        return loss, {
            "mean_group_reward": group.rewards.detach().mean().item(),
            "mean_advantage_magnitude": advantages.detach().abs().mean().item(),
            "kl_divergence": kl_divergence.detach().item(),
            "clipped_policy_loss": clipped_policy_loss.detach().item(),
        }


def run_grpo(
    model: nn.Module,
    prompts: Sequence[str],
    reward_fn: RewardFunction,
    *,
    tokenizer: Any,
    ref_model: nn.Module | None = None,
    group_size: int = 4,
    beta: float = 0.04,
    clip_epsilon: float = 0.2,
    learning_rate: float = 1e-5,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    advantage_epsilon: float = 1e-6,
    num_steps: int | None = None,
    optimizer: Optimizer | None = None,
    device: torch.device | str = torch.device("cpu"),
    display_config: Any | None = None,
    **model_kwargs: Any,
) -> list[float]:
    """Run GRPO directly from text prompts and a user-defined reward function."""

    return GRPOTrainer(
        model,
        prompts,
        reward_fn,
        tokenizer=tokenizer,
        ref_model=ref_model,
        group_size=group_size,
        beta=beta,
        clip_epsilon=clip_epsilon,
        learning_rate=learning_rate,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        advantage_epsilon=advantage_epsilon,
        num_steps=num_steps,
        optimizer=optimizer,
        device=device,
        display_config=display_config,
        model_kwargs=model_kwargs,
    ).train()


def group_relative_advantages(rewards: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Normalize one completion group's rewards using population standard deviation."""

    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("rewards must be a one-dimensional group with at least two values.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    rewards = rewards.float()
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + epsilon)


def _encode_prompt(tokenizer: Any, prompt: str, *, max_tokens: int | None = None) -> list[int]:
    """Format a user turn and reserve room for a generated assistant reply."""

    chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(chat_template):
        formatted_prompt = chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        # Some HF-compatible tokenizer wrappers return formatted text despite
        # tokenize=True. Normalize both API variants before tensor creation.
        if isinstance(formatted_prompt, str):
            if not hasattr(tokenizer, "encode"):
                raise TypeError("String chat templates require tokenizer.encode().")
            prompt_ids = list(tokenizer.encode(formatted_prompt, add_special_tokens=False))
        elif isinstance(formatted_prompt, Tensor):
            prompt_ids = formatted_prompt.reshape(-1).tolist()
        else:
            prompt_ids = list(formatted_prompt)
    elif hasattr(tokenizer, "encode"):
        prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    else:
        raise TypeError("tokenizer must provide encode() or apply_chat_template().")
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, Integral)
        for token_id in prompt_ids
    ):
        raise TypeError("Formatted GRPO prompts must tokenize to a sequence of integer token ids.")
    prompt_ids = [int(token_id) for token_id in prompt_ids]
    if max_tokens is not None:
        if max_tokens <= 0:
            raise ValueError("Model context limit leaves no room for GRPO completion tokens.")
        prompt_ids = prompt_ids[-max_tokens:]
    if not prompt_ids:
        raise ValueError("Each prompt must tokenize to at least one token.")
    return prompt_ids


def _prompt_token_budget(model: nn.Module, max_new_tokens: int) -> int | None:
    """Reserve completion capacity under an HF model's finite context length."""

    config = getattr(model, "config", None)
    context_limit = getattr(config, "max_position_embeddings", None)
    if isinstance(context_limit, int) and context_limit > 0:
        return context_limit - max_new_tokens
    return None


def _generate(
    model: nn.Module,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    pad_token_id: int,
    device: torch.device,
) -> Tensor:
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    # GRPO prompts have no padding, but HF generation cannot safely infer that
    # when pad_token_id equals eos_token_id (the common SmolLM2 setup).
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        min_new_tokens=1,
        pad_token_id=pad_token_id,
    )
    if not isinstance(generated, Tensor) or generated.ndim != 2 or generated.size(0) != 1:
        raise TypeError("model.generate() must return token ids with shape (1, sequence_length).")
    return generated[0].detach()


def _decode_completion(tokenizer: Any, completion_ids: Tensor) -> str:
    if not hasattr(tokenizer, "decode"):
        raise TypeError("tokenizer must provide decode(token_ids, skip_special_tokens=True).")
    return str(tokenizer.decode(completion_ids.tolist(), skip_special_tokens=True))


def _pad_token_id(tokenizer: Any) -> int:
    for name in ("pad_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, name, None)
        if isinstance(token_id, int) and token_id >= 0:
            return token_id
    return 0


def _pack_completions(
    completions: Sequence[Tensor], *, prompt_length: int, pad_token_id: int
) -> tuple[Tensor, Tensor]:
    max_length = max(completion.size(0) for completion in completions)
    input_ids = torch.full(
        (len(completions), max_length), pad_token_id, dtype=torch.long, device=completions[0].device
    )
    completion_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, completion in enumerate(completions):
        input_ids[index, : completion.size(0)] = completion
        completion_mask[index, prompt_length : completion.size(0)] = True
    return input_ids, completion_mask


def _sequence_log_probabilities(logits: Tensor, input_ids: Tensor, completion_mask: Tensor) -> Tensor:
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise ValueError("Model logits must have shape (batch, sequence, vocab_size).")
    if completion_mask.shape != input_ids.shape:
        raise ValueError("completion_mask must match input_ids.")
    token_mask = completion_mask[:, 1:]
    if not torch.all(token_mask.any(dim=1)):
        raise ValueError("Every completion must contain at least one predicted token.")
    token_log_probs = F.log_softmax(logits[:, :-1, :], dim=-1).gather(
        dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    return (token_log_probs * token_mask).sum(dim=-1)


def _completion_kl_divergence(
    policy_logits: Tensor, reference_logits: Tensor, completion_mask: Tensor
) -> Tensor:
    if policy_logits.shape != reference_logits.shape:
        raise ValueError("Policy and reference logits must have the same shape.")
    token_mask = completion_mask[:, 1:].to(dtype=policy_logits.dtype)
    policy_log_probs = F.log_softmax(policy_logits[:, :-1, :], dim=-1)
    reference_log_probs = F.log_softmax(reference_logits[:, :-1, :], dim=-1)
    reference_probabilities = reference_log_probs.exp()
    token_kl = (reference_probabilities * (reference_log_probs - policy_log_probs)).sum(dim=-1)
    return (token_kl * token_mask).sum() / token_mask.sum().clamp_min(1)


def _extract_logits(model_output: Any) -> Tensor:
    if isinstance(model_output, Tensor):
        return model_output
    if hasattr(model_output, "logits") and isinstance(model_output.logits, Tensor):
        return model_output.logits
    if isinstance(model_output, Mapping) and isinstance(model_output.get("logits"), Tensor):
        return model_output["logits"]
    raise TypeError("Model output must be logits or expose a Tensor 'logits' attribute.")


def _make_display(
    display_config: Any | None, total_steps: int | None
) -> TrainingDisplay | AbstractContextManager[None]:
    return (
        TrainingDisplay(display_config, total_steps=total_steps, loss_label="grpo_loss")
        if display_config
        else _NullDisplay()
    )


class _NullDisplay(AbstractContextManager[None]):
    """No-op display for direct script usage without a Picotron run config."""

    def update(self, **_: Any) -> None:
        pass

    def __enter__(self) -> "_NullDisplay":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None
