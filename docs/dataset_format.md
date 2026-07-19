# Dataset format for Picotron SFT and DPO

Picotron does not own dataset preparation. Use the Hugging Face `datasets`
library (or any `torch.utils.data.Dataset`) to prepare the records below, then
pass the dataset to the trainer.

## SFT

`PicotronSFTTrainer` expects a map-style dataset whose examples contain a text
field. The default field name is `text`; change it with
`PicotronSFTConfig(dataset_text_field="...")`.

```python
from datasets import Dataset

train_dataset = Dataset.from_dict({
    "text": [
        "user: What is 2 + 2?\nassistant: 4",
        "user: Name a color.\nassistant: Blue.",
    ]
})
```

The trainer tokenizes and right-pads this text. Its current baseline masks
padding only; it does not automatically apply assistant-only masking or a chat
template. Apply your model's chat template while constructing `text` if needed.

The lower-level `run_sft()` API remains available for pre-tokenized mappings:
`{"input_ids": Tensor, "labels": Tensor}` (and optional
`"attention_mask"`). Labels use `-100` for ignored tokens.

## DPO

`PicotronDPOTrainer` and `run_dpo()` expect prompt/preference triples. A list
of tuples works:

```python
preferences = [
    ("user: Name a color.\nassistant: ", "Blue.", "Seven."),
]
```

or a dataset with `prompt`, `chosen`, and `rejected` string columns:

```python
preferences = Dataset.from_dict({
    "prompt": ["user: Name a color.\nassistant: "],
    "chosen": ["Blue."],
    "rejected": ["Seven."],
})
```

The prompt must already include any chat-template generation prefix. `chosen`
and `rejected` must be completion-only strings. Picotron tokenizes them,
masks prompt tokens, and computes the standard DPO objective.

## Current trainer limits

The TRL-style Picotron wrappers currently use AdamW with a constant learning
rate. They reject nonzero warmup, non-constant schedulers, and gradient
accumulation above one rather than silently changing optimization behavior.
Use the direct APIs for existing custom/pre-tokenized loaders.
