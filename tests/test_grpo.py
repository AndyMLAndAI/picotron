"""CPU correctness checks for group-relative policy optimization."""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from picotron_grpo import GRPOTrainer, group_relative_advantages, run_grpo
from picotron_grpo.grpo_trainer import _encode_prompt, _generate


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del text, add_special_tokens
        return [1]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return "X" if token_ids == [2] else "Y"


class _TinyGenerativePolicy(nn.Module):
    """A one-step categorical causal LM with a standard ``generate`` method."""

    def __init__(self) -> None:
        super().__init__()
        self.next_token_logits = nn.Parameter(torch.tensor([-12.0, -12.0, 0.0, 0.0]))

    def forward(self, input_ids: Tensor, **_: object) -> Tensor:
        return self.next_token_logits.view(1, 1, -1).expand(
            input_ids.size(0), input_ids.size(1), -1
        )

    def generate(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        min_new_tokens: int,
        pad_token_id: int,
    ) -> Tensor:
        del do_sample, min_new_tokens, pad_token_id
        if attention_mask is not None:
            assert torch.equal(attention_mask, torch.ones_like(input_ids))
        generated = input_ids
        for _ in range(max_new_tokens):
            probabilities = torch.softmax(self.next_token_logits / temperature, dim=0)
            next_token = torch.multinomial(probabilities, input_ids.size(0), replacement=True)
            generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)
        return generated


def _reward(_: str, completion: str) -> float:
    return float(completion == "X")


def _reward_rate(model: _TinyGenerativePolicy, samples: int = 256) -> float:
    prompt = torch.tensor([[1]], dtype=torch.long)
    rewarded = 0
    with torch.no_grad():
        for _ in range(samples):
            rewarded += int(model.generate(
                prompt,
                max_new_tokens=1,
                do_sample=True,
                temperature=1.0,
                min_new_tokens=1,
                pad_token_id=0,
            )[0, -1].item() == 2)
    return rewarded / samples


def test_group_relative_advantages_match_population_standard_deviation() -> None:
    rewards = torch.tensor([1.0, 3.0, 5.0])

    advantages = group_relative_advantages(rewards, epsilon=1e-6)

    expected = (rewards - 3.0) / (torch.sqrt(torch.tensor(8.0 / 3.0)) + 1e-6)
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(advantages.mean(), torch.tensor(0.0), atol=1e-6, rtol=0)


def test_grpo_keeps_reference_weights_frozen() -> None:
    torch.manual_seed(11)
    policy = _TinyGenerativePolicy()
    reference = copy.deepcopy(policy)
    reference_before = {name: value.detach().clone() for name, value in reference.state_dict().items()}

    trainer = GRPOTrainer(
        policy,
        ["prompt"],
        _reward,
        tokenizer=_Tokenizer(),
        ref_model=reference,
        group_size=4,
        beta=0.04,
        learning_rate=0.1,
        max_new_tokens=1,
        num_steps=20,
    )
    losses = trainer.train()

    assert len(losses) == 20
    for name, before in reference_before.items():
        torch.testing.assert_close(reference.state_dict()[name], before, rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in reference.parameters())


def test_grpo_increases_directional_reward_rate() -> None:
    torch.manual_seed(23)
    policy = _TinyGenerativePolicy()
    tokenizer = _Tokenizer()
    before = _reward_rate(policy)

    losses = run_grpo(
        policy,
        ["prompt"],
        _reward,
        tokenizer=tokenizer,
        group_size=4,
        beta=0.0,
        clip_epsilon=0.2,
        learning_rate=0.1,
        max_new_tokens=1,
        num_steps=150,
    )
    after = _reward_rate(policy)

    print(f"GRPO rewarded completion rate: before={before:.3f}, after={after:.3f}")
    assert len(losses) == 150
    assert after > before + 0.15


class _ChatTokenizer(_Tokenizer):
    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert tokenize and add_generation_prompt
        assert messages == [{"role": "user", "content": "What is 6 times 7?"}]
        return [9, 1]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return "42" if token_ids else ""


class _StringChatTokenizer(_ChatTokenizer):
    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert tokenize and add_generation_prompt
        assert messages == [{"role": "user", "content": "What is 6 times 7?"}]
        return "user: What is 6 times 7?\nassistant:"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text.endswith("assistant:")
        assert not add_special_tokens
        return [9, 1]


def test_gsm8k_style_generation_uses_chat_template_and_is_non_empty() -> None:
    """GRPO must leave room for a textual assistant completion, not immediate EOS."""

    tokenizer = _ChatTokenizer()
    prompt_ids = _encode_prompt(tokenizer, "What is 6 times 7?", max_tokens=4)
    completion_ids = _generate(
        _TinyGenerativePolicy(),
        prompt_ids,
        max_new_tokens=1,
        temperature=1.0,
        pad_token_id=0,
        device=torch.device("cpu"),
    )[len(prompt_ids) :]
    completion = tokenizer.decode(completion_ids.tolist(), skip_special_tokens=True)
    assert completion
    assert len(completion) <= 16


def test_string_chat_template_is_tokenized_before_generation() -> None:
    """Prevent formatted chat text from reaching torch.tensor in _generate."""

    tokenizer = _StringChatTokenizer()
    prompt_ids = _encode_prompt(tokenizer, "What is 6 times 7?", max_tokens=4)
    assert prompt_ids == [9, 1]
    assert all(isinstance(token_id, int) for token_id in prompt_ids)
    generated = _generate(
        _TinyGenerativePolicy(),
        prompt_ids,
        max_new_tokens=1,
        temperature=1.0,
        pad_token_id=0,
        device=torch.device("cpu"),
    )
    assert generated.ndim == 1
