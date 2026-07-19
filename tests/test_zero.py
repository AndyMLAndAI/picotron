"""CPU/Gloo parity checks for ZeRO Stage 1 and Stage 2."""

from dataclasses import replace
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from picotron.config.config import load_config
from picotron.models.picotron_decoder import PicotronDecoderModel
from picotron.parallel.ddp import cleanup_distributed, initialize_distributed
from picotron.training.train_loop import train


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_rank(
    rank: int,
    world_size: int,
    port: int,
    zero_stage: int,
    config_path: str,
    output_dir: str,
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    loaded_config = load_config(config_path)
    config = replace(
        loaded_config,
        parallelism=replace(
            loaded_config.parallelism, dp=world_size, zero_stage=zero_stage
        ),
    )
    torch.manual_seed(41)
    model = PicotronDecoderModel(config)
    batch = (
        torch.arange(
            config.tokens.micro_batch_size * config.tokens.sequence_length,
            dtype=torch.long,
        )
        .reshape(config.tokens.micro_batch_size, config.tokens.sequence_length)
        .add(rank)
        .remainder(config.model.model_config.vocab_size)
    )
    initialize_distributed(backend="gloo")
    train(model, [batch, batch, batch], config, max_steps=3)
    torch.save(model.state_dict(), Path(output_dir) / f"stage_{zero_stage}_rank_{rank}.pt")
    dist.barrier()
    cleanup_distributed()


def _run_stage(zero_stage: int, tmp_path: Path) -> dict[str, torch.Tensor]:
    config_path = Path(__file__).resolve().parents[1] / "src/picotron/config/picotron_decoder.yaml"
    context = mp.get_context("spawn")
    port = _free_port()
    processes = [
        context.Process(
            target=_run_rank,
            args=(rank, 2, port, zero_stage, str(config_path), str(tmp_path)),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
        assert process.exitcode == 0
    rank_zero = torch.load(tmp_path / f"stage_{zero_stage}_rank_0.pt", map_location="cpu")
    rank_one = torch.load(tmp_path / f"stage_{zero_stage}_rank_1.pt", map_location="cpu")
    for name in rank_zero:
        torch.testing.assert_close(rank_zero[name], rank_one[name], rtol=0, atol=0)
    return rank_zero


def test_zero_matches_plain_ddp_math(tmp_path: Path) -> None:
    ddp_weights = _run_stage(0, tmp_path)
    zero_one_weights = _run_stage(1, tmp_path)
    zero_two_weights = _run_stage(2, tmp_path)

    for name, ddp_weight in ddp_weights.items():
        torch.testing.assert_close(ddp_weight, zero_one_weights[name], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(ddp_weight, zero_two_weights[name], rtol=1e-6, atol=1e-6)
