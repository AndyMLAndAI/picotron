"""CPU/Gloo proxies for distributed loss scaling and ZeRO checkpoint resume."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from picotron.config.config import load_config
from picotron.models.picotron_decoder import PicotronDecoderModel
from picotron.parallel.ddp import cleanup_distributed, initialize_distributed, wrap_model
from picotron.parallel.zero import DistributedGradScaler, ZeroOptimizer
from picotron.serialize.checkpoint import load_checkpoint
from picotron.training.train_loop import train


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_handle:
        socket_handle.bind(("127.0.0.1", 0))
        return int(socket_handle.getsockname()[1])


def _set_rank_environment(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )


def _rank_batch(config: object, rank: int) -> torch.Tensor:
    return (
        torch.arange(
            config.tokens.micro_batch_size * config.tokens.sequence_length,
            dtype=torch.long,
        )
        .reshape(config.tokens.micro_batch_size, config.tokens.sequence_length)
        .add(rank)
        .remainder(config.model.model_config.vocab_size)
    )


def _loss(model: torch.nn.Module, batch: torch.Tensor, vocab_size: int) -> torch.Tensor:
    logits = model(batch)
    return F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), batch[:, 1:].reshape(-1))


def _loss_scale_worker(rank: int, world_size: int, port: int, config_path: str, output: str) -> None:
    _set_rank_environment(rank, world_size, port)
    loaded = load_config(config_path)
    config = replace(loaded, parallelism=replace(loaded.parallelism, dp=world_size, zero_stage=1))
    initialize_distributed(backend="gloo")
    batch = _rank_batch(config, rank)

    def run(*, use_scaler: bool) -> tuple[list[float], dict[str, torch.Tensor]]:
        torch.manual_seed(91)
        model = wrap_model(PicotronDecoderModel(config), initialize_distributed(), device="cpu")
        optimizer = ZeroOptimizer(model.parameters(), learning_rate=1e-3, stage=1)
        scaler = DistributedGradScaler("cpu", init_scale=128.0, growth_interval=2) if use_scaler else None
        losses: list[float] = []
        for _ in range(4):
            optimizer.zero_grad()
            loss = _loss(model, batch, config.model.model_config.vocab_size)
            if scaler is None:
                optimizer.backward(loss, model)
                optimizer.step()
            else:
                optimizer.backward(scaler.scale(loss), model)
                assert scaler.step(optimizer, model)
            losses.append(loss.detach().item())
        return losses, {name: value.detach().cpu() for name, value in model.module.state_dict().items()}

    scaled_losses, scaled_weights = run(use_scaler=True)
    baseline_losses, baseline_weights = run(use_scaler=False)
    torch.manual_seed(92)
    overflow_model = wrap_model(PicotronDecoderModel(config), initialize_distributed(), device="cpu")
    overflow_optimizer = ZeroOptimizer(overflow_model.parameters(), learning_rate=1e-3, stage=1)
    overflow_scaler = DistributedGradScaler("cpu", init_scale=128.0)
    overflow_optimizer.zero_grad()
    overflow_loss = _loss(overflow_model, batch, config.model.model_config.vocab_size)
    overflow_optimizer.backward(overflow_scaler.scale(overflow_loss), overflow_model)
    if rank == 0:
        next(parameter for parameter in overflow_model.parameters()).grad.fill_(float("inf"))
    overflow_skipped = not overflow_scaler.step(overflow_optimizer, overflow_model)
    torch.save(
        {
            "scaled_losses": scaled_losses,
            "baseline_losses": baseline_losses,
            "scaled_weights": scaled_weights,
            "baseline_weights": baseline_weights,
            "overflow_skipped": overflow_skipped,
        },
        Path(output) / f"loss_scale_rank_{rank}.pt",
    )
    dist.barrier()
    cleanup_distributed()


def _checkpoint_worker(rank: int, world_size: int, port: int, config_path: str, output: str) -> None:
    _set_rank_environment(rank, world_size, port)
    loaded = load_config(config_path)
    checkpoint_path = Path(output) / "zero_resume.safetensors"
    config = replace(
        loaded,
        parallelism=replace(loaded.parallelism, dp=world_size, zero_stage=1),
        checkpoints=replace(
            loaded.checkpoints,
            checkpoint_interval=3,
            checkpoints_path=str(checkpoint_path),
            save_final_state=False,
        ),
        logging=replace(loaded.logging, file_logging=False),
    )
    torch.manual_seed(47)
    original_model = PicotronDecoderModel(config)
    batch = _rank_batch(config, rank)
    first_losses = train(original_model, [batch], config, max_steps=3, device="cpu")
    saved_weights = {name: value.detach().clone() for name, value in original_model.state_dict().items()}

    fresh_model = wrap_model(PicotronDecoderModel(config), initialize_distributed(), device="cpu")
    fresh_optimizer = ZeroOptimizer(fresh_model.parameters(), learning_rate=1e-3, stage=1)
    resumed_step = load_checkpoint(fresh_model, fresh_optimizer, checkpoint_path)
    exact_weights = all(
        torch.equal(value, fresh_model.module.state_dict()[name])
        for name, value in saved_weights.items()
    )

    resumed_model = PicotronDecoderModel(config)
    resumed_losses = train(
        resumed_model,
        [batch],
        config,
        max_steps=3,
        resume_from=checkpoint_path,
        device="cpu",
    )
    torch.save(
        {
            "first_losses": first_losses,
            "resumed_losses": resumed_losses,
            "resumed_step": resumed_step,
            "exact_weights": exact_weights,
        },
        Path(output) / f"checkpoint_rank_{rank}.pt",
    )
    dist.barrier()
    cleanup_distributed()


def _spawn(worker: object, tmp_path: Path, config_path: Path) -> None:
    context = mp.get_context("spawn")
    port = _free_port()
    processes = [
        context.Process(target=worker, args=(rank, 2, port, str(config_path), str(tmp_path)))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(90)
        assert process.exitcode == 0


def test_loss_scale_proxy_matches_unscaled_zero_math(tmp_path: Path) -> None:
    """CPU cannot execute CUDA fp16, so exercise the exact scale/unscale protocol."""

    config_path = Path(__file__).resolve().parents[1] / "src/picotron/config/picotron_decoder.yaml"
    _spawn(_loss_scale_worker, tmp_path, config_path)
    result = torch.load(tmp_path / "loss_scale_rank_0.pt", map_location="cpu")
    rank_one = torch.load(tmp_path / "loss_scale_rank_1.pt", map_location="cpu")
    assert result["scaled_losses"][-1] < result["scaled_losses"][0]
    assert result["overflow_skipped"] and rank_one["overflow_skipped"]
    for scaled, baseline in zip(result["scaled_losses"], result["baseline_losses"], strict=True):
        assert abs(scaled - baseline) < 1e-5
    for name, value in result["scaled_weights"].items():
        torch.testing.assert_close(value, result["baseline_weights"][name], rtol=1e-5, atol=1e-6)


def test_zero_checkpoint_resume_uses_rank_local_optimizer_shards(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "src/picotron/config/picotron_decoder.yaml"
    _spawn(_checkpoint_worker, tmp_path, config_path)

    rank_zero = torch.load(tmp_path / "checkpoint_rank_0.pt", map_location="cpu")
    rank_one = torch.load(tmp_path / "checkpoint_rank_1.pt", map_location="cpu")
    assert rank_zero["resumed_step"] == 3
    assert rank_zero["exact_weights"] and rank_one["exact_weights"]
    assert rank_zero["resumed_losses"][0] <= rank_zero["first_losses"][-1] + 0.25
    assert rank_zero["resumed_losses"][-1] < rank_zero["resumed_losses"][0]
    assert (tmp_path / "zero_resume.safetensors").exists()
    assert (tmp_path / "zero_resume.optimizer.pt").exists()
    assert (tmp_path / "zero_resume.optimizer.rank0.pt").exists()
    assert (tmp_path / "zero_resume.optimizer.rank1.pt").exists()
