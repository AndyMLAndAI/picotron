"""CPU/Gloo DDP synchronization tests."""

import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.optim import AdamW

from picotron.config.config import load_config
from picotron.models.toy_model import ToyDecoderModel
from picotron.parallel.ddp import cleanup_distributed, initialize_distributed, wrap_model


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ddp_worker(
    rank: int,
    world_size: int,
    port: int,
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
    config = load_config(config_path)
    torch.manual_seed(23)
    model = ToyDecoderModel(config)
    info = initialize_distributed(backend="gloo")
    model = wrap_model(model, info, device="cpu")
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    input_ids = (
        torch.arange(config.batch_size * config.max_seq_len, dtype=torch.long)
        .reshape(config.batch_size, config.max_seq_len)
        .add(rank)
        .remainder(config.vocab_size)
    )
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, config.vocab_size), input_ids[:, 1:].reshape(-1)
    )
    loss.backward()
    optimizer.step()
    torch.save(
        {name: value.detach().cpu() for name, value in model.module.state_dict().items()},
        Path(output_dir) / f"rank_{rank}.pt",
    )
    dist.barrier()
    cleanup_distributed()


def test_single_process_fallback() -> None:
    info = initialize_distributed()
    assert info.world_size == 1
    assert not info.is_distributed


def test_two_rank_cpu_gradients_synchronize(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "src/picotron/config/toy_model.yaml"
    port = _free_port()
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_ddp_worker,
            args=(rank, 2, port, str(config_path), str(tmp_path)),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
        assert process.exitcode == 0

    rank_zero = torch.load(tmp_path / "rank_0.pt", map_location="cpu")
    rank_one = torch.load(tmp_path / "rank_1.pt", map_location="cpu")
    assert rank_zero.keys() == rank_one.keys()
    for name in rank_zero:
        torch.testing.assert_close(rank_zero[name], rank_one[name], rtol=0, atol=0)

