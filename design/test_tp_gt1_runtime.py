"""CPU checks against installed Ascend classes; collectives are mocked.

Run in the reviewed vLLM/vllm-ascend Python environment, alongside existing
vocabulary UTs so the repository's UT setup is loaded. This does not execute
NPU kernels, HCCL, model inference or graph capture.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm_ascend.ascend_config import FinegrainedTPConfig
from vllm_ascend.ops import vocab_parallel_embedding as vocab


CASES = [(tp, dp, size) for tp in (2, 4, 8) for dp, size in ((4, 2), (4, 4), (8, 2), (8, 4), (8, 8))]
V, H, CAPACITY = 64, 4, 3


def no_native_tp(*args, **kwargs):
    raise AssertionError("Fine-grained vocabulary must not also reduce/gather over native TP")


def configure(monkeypatch, tp, dp, size, tp_rank, dp_rank):
    group = SimpleNamespace(world_size=size, rank_in_group=dp_rank % size, device_group=object())
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, data_parallel_size=dp, prefill_context_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=CAPACITY),
        model_config=SimpleNamespace(is_moe=True),
    )
    FinegrainedTPConfig(embedding_tensor_parallel_size=size, lmhead_tensor_parallel_size=size)._validate_preconditions(cfg)
    values = {
        "get_tp_group": SimpleNamespace(world_size=tp, rank_in_group=tp_rank),
        "get_pcp_group": SimpleNamespace(world_size=1, rank_in_group=0),
        "get_embed_tp_group": group,
        "get_lmhead_tp_group": group,
        "get_current_vllm_config": cfg,
        "get_potential_max_tokens": CAPACITY,
        "enable_pcp_embedding_lmhead_weight_sharding": False,
        "embedding_tp_enable": True,
        "lmhead_tp_enable": True,
    }
    for name, value in values.items():
        monkeypatch.setattr(vocab, name, lambda value=value: value)
    monkeypatch.setattr(torch.ops.vllm, "all_reduce", no_native_tp)
    return group


@pytest.mark.parametrize("tp,dp,size", CASES)
def test_embedding_native_tp_replaced_by_dp_axis_shards(monkeypatch, tp, dp, size):
    full_weight = torch.arange(V * H, dtype=torch.float32).reshape(V, H) / 16
    for tp_rank in range(tp):
        for dp_rank in range(dp):
            group = configure(monkeypatch, tp, dp, size, tp_rank, dp_rank)
            layer = vocab.AscendVocabParallelEmbedding(V, H, padding_size=8, prefix="model.embed_tokens")
            layer.weight_loader(layer.weight, full_weight)
            assert layer.parallel_mode is vocab.VocabParallelMode.FINE_GRAINED
            assert layer.weight.shape == (V // size, H)  # NOT V/(tp*size)
            torch.testing.assert_close(layer.weight, full_weight.chunk(size)[dp_rank % size])
            base = dp_rank // size * size
            for step in range(3):
                ids = [torch.tensor([(r * 7 + k) % V for k in range((r + step) % 4)], dtype=torch.long) for r in range(dp)]
                padded = torch.stack([F.pad(ids[r], (0, CAPACITY - len(ids[r]))) for r in range(base, base + size)])
                gathered = padded.flatten()

                def gather(tensor, dim):
                    assert dim == 0
                    torch.testing.assert_close(tensor, padded[group.rank_in_group])
                    return gathered

                def scatter(tensor, dim):
                    assert dim == 0
                    start = layer.shard_indices.org_vocab_start_index
                    end = layer.shard_indices.org_vocab_end_index
                    local = full_weight[gathered] * ((gathered >= start) & (gathered < end)).unsqueeze(-1)
                    torch.testing.assert_close(tensor, local)
                    return full_weight[padded[group.rank_in_group]]

                group.all_gather, group.reduce_scatter = gather, scatter
                with torch.inference_mode():
                    actual = layer(ids[dp_rank])
                torch.testing.assert_close(actual, full_weight[ids[dp_rank]])


@pytest.mark.parametrize("tp,dp,size", CASES)
def test_lmhead_native_tp_replaced_by_dp_axis_shards(monkeypatch, tp, dp, size):
    full_weight = torch.arange(V * H, dtype=torch.float32).reshape(V, H) / 16
    processor = object.__new__(vocab.AscendLogitsProcessor)
    nn.Module.__init__(processor)
    processor.org_vocab_size, processor.head_dtype = V, None
    processor._gather_logits = no_native_tp
    for tp_rank in range(tp):
        for dp_rank in range(dp):
            group = configure(monkeypatch, tp, dp, size, tp_rank, dp_rank)
            layer = vocab.AscendParallelLMHead(V, H, padding_size=8, prefix="lm_head")
            layer.weight_loader(layer.weight, full_weight)
            assert layer.weight.shape == (V // size, H)
            torch.testing.assert_close(layer.weight, full_weight.chunk(size)[dp_rank % size])
            base = dp_rank // size * size
            for step in range(3):
                # An owner may have no sampled rows; it still feeds the fixed capacity.
                counts = [(rank + step) % 4 for rank in range(dp)]
                hidden = [F.pad(torch.arange(counts[rank] * H, dtype=torch.float32).reshape(-1, H) + rank,
                                (0, 0, 0, CAPACITY - counts[rank])) for rank in range(dp)]
                gathered = torch.cat(hidden[base:base + size])

                def gather(tensor, dim):
                    assert dim == 0
                    torch.testing.assert_close(tensor, hidden[dp_rank])
                    return gathered

                def all_to_all(output, tensor, group=None):
                    torch.testing.assert_close(tensor, F.linear(gathered, layer.weight).view(size, CAPACITY, V // size))
                    owner_logits = F.linear(hidden[dp_rank], full_weight)
                    output.copy_(owner_logits.view(CAPACITY, size, V // size).permute(1, 0, 2))

                group.all_gather = gather
                monkeypatch.setattr(vocab.dist, "all_to_all_single", all_to_all)
                with torch.inference_mode():
                    actual = processor._get_logits(hidden[dp_rank], layer)
                torch.testing.assert_close(actual, F.linear(hidden[dp_rank], full_weight))
                assert actual[:counts[dp_rank]].shape == (counts[dp_rank], V)


@pytest.mark.parametrize("tp", (2, 4, 8))
@pytest.mark.parametrize("field", ("oproj_tensor_parallel_size", "mlp_tensor_parallel_size"))
def test_current_config_explicitly_rejects_o_mlp_with_native_tp(tp, field):
    config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=tp))
    with pytest.raises(AssertionError, match="require tensor_parallel_size == 1"):
        FinegrainedTPConfig(**{field: 2})._validate_preconditions(config)
