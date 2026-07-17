# Picotron

Picotron is a from-scratch distributed LLM pretraining framework.

## Current scaffold

This phase provides strict YAML configuration loading, a CPU-testable toy decoder-only model, and safe hardware capability detection. Training, data loading, checkpointing, and distributed execution are intentionally not implemented yet.

## CPU checks

From this directory, run:

```powershell
python -m pytest tests/test_model.py tests/test_hardware.py
```

