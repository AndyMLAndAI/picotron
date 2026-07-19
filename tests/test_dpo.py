"""CPU correctness checks for Direct Preference Optimization."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from picotron.config.config import load_config
from picotron.models.toy_model import ToyDecoderModel
from picotron_dpo.data import PreferenceDataset, collate_preference_batch
from picotron_dpo.dpo_trainer import DPOTrainer, _sequence_log_probability


class _PreferenceTokenizer:
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        token_ids = {"P": 1, "A": 2, "B": 3}
        return [token_ids[character] for character in text]


def test_dpo_increases_chosen_margin_and_keeps_reference_frozen() -> None:
    config_path = Path(__file__).resolve().parents[1] / "src/picotron/config/toy_model.yaml"
    loaded_config = load_config(config_path)
    config = replace(
        loaded_config,
        tokens=replace(loaded_config.tokens, sequence_length=4, micro_batch_size=4),
    )
    tokenizer = _PreferenceTokenizer()
    preferences = [("P", "A", "B")] * 256
    dataset = PreferenceDataset(preferences, tokenizer, max_length=config.tokens.sequence_length)
    data_loader = DataLoader(
        dataset,
        batch_size=config.tokens.micro_batch_size,
        collate_fn=lambda examples: collate_preference_batch(examples, pad_token_id=0),
    )

    torch.manual_seed(41)
    model = ToyDecoderModel(config)
    trainer = DPOTrainer(
        model,
        data_loader,
        beta=0.1,
        optimizer=AdamW(model.parameters(), lr=0.02),
        num_steps=64,
    )
    reference_weights = {
        name: parameter.detach().clone() for name, parameter in trainer.ref_model.state_dict().items()
    }
    batch = next(iter(data_loader))

    with torch.no_grad():
        initial_chosen, initial_rejected = _log_probabilities(trainer.model, batch)
        reference_chosen, reference_rejected = _log_probabilities(trainer.ref_model, batch)
        initial_margin = _margin(trainer.model, batch) - _margin(trainer.ref_model, batch)
    print(
        "before_dpo "
        f"policy_chosen={initial_chosen:.6f} policy_rejected={initial_rejected:.6f} "
        f"reference_chosen={reference_chosen:.6f} reference_rejected={reference_rejected:.6f} "
        f"relative_margin={initial_margin.item():.6f}"
    )
    losses = trainer.train()
    with torch.no_grad():
        final_chosen, final_rejected = _log_probabilities(trainer.model, batch)
        final_margin = _margin(trainer.model, batch) - _margin(trainer.ref_model, batch)

    reference_unchanged = all(
        torch.equal(trainer.ref_model.state_dict()[name], expected)
        for name, expected in reference_weights.items()
    )
    print(
        "after_dpo "
        f"policy_chosen={final_chosen:.6f} policy_rejected={final_rejected:.6f} "
        f"reference_chosen={reference_chosen:.6f} reference_rejected={reference_rejected:.6f} "
        f"relative_margin={final_margin.item():.6f} "
        f"reference_weights_unchanged={reference_unchanged}"
    )

    assert len(losses) == 64
    assert final_margin.item() > initial_margin.item()
    assert final_margin.item() > 0.0
    assert all(not parameter.requires_grad for parameter in trainer.ref_model.parameters())
    assert reference_unchanged


def _margin(model: ToyDecoderModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    chosen = _sequence_log_probability(model(batch["chosen_input_ids"]), batch["chosen_labels"])
    rejected = _sequence_log_probability(model(batch["rejected_input_ids"]), batch["rejected_labels"])
    return (chosen - rejected).mean()


def _log_probabilities(model: ToyDecoderModel, batch: dict[str, torch.Tensor]) -> tuple[float, float]:
    chosen = _sequence_log_probability(model(batch["chosen_input_ids"]), batch["chosen_labels"])
    rejected = _sequence_log_probability(model(batch["rejected_input_ids"]), batch["rejected_labels"])
    return chosen.mean().item(), rejected.mean().item()
