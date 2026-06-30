import einops
import pytest
import torch
import torch.nn as nn
from transformers.utils import is_bitsandbytes_available

from transformer_lens.components import Attention
from transformer_lens.config import HookedTransformerConfig
from transformer_lens.utilities.attention import complex_attn_linear

if is_bitsandbytes_available():
    from bitsandbytes.nn.modules import Params4bit


def test_attention_hooked_transformer_config():
    cfg = HookedTransformerConfig(
        n_layers=12,
        d_model=512,
        n_ctx=1024,
        d_head=64,
        n_heads=8,
        load_in_4bit=False,
        dtype=torch.float32,
        act_fn="relu",
    )
    attn = Attention(cfg)
    assert attn.cfg == cfg
    assert attn.cfg.n_layers == 12
    assert attn.cfg.d_model == 512
    assert attn.cfg.n_ctx == 1024
    assert attn.cfg.d_head == 64
    assert attn.cfg.n_heads == 8
    assert attn.cfg.load_in_4bit == False
    assert attn.cfg.dtype == torch.float32
    assert attn.cfg.act_fn == "relu"

    assert isinstance(attn.W_K, nn.Parameter)
    assert isinstance(attn.W_V, nn.Parameter)
    assert attn.W_K.shape == (cfg.n_heads, cfg.d_model, cfg.d_head)
    assert attn.W_V.shape == (cfg.n_heads, cfg.d_model, cfg.d_head)

    assert attn.b_K.shape == (cfg.n_heads, cfg.d_head)
    assert attn.b_V.shape == (cfg.n_heads, cfg.d_head)
    assert torch.all(attn.b_K == 0)
    assert torch.all(attn.b_V == 0)


@pytest.mark.skipif(not is_bitsandbytes_available(), reason="bitsandbytes is not available")
def test_attention_load_in_4bit():
    cfg = HookedTransformerConfig(
        n_layers=12,
        d_model=512,
        n_ctx=1024,
        d_head=64,
        n_heads=8,
        load_in_4bit=True,
        dtype=torch.float32,
        act_fn="relu",
    )
    attn = Attention(cfg)
    assert attn.cfg == cfg
    assert attn.cfg.n_layers == 12
    assert attn.cfg.d_model == 512
    assert attn.cfg.n_ctx == 1024
    assert attn.cfg.d_head == 64
    assert attn.cfg.n_heads == 8
    assert attn.cfg.load_in_4bit == True
    assert attn.cfg.dtype == torch.float32
    assert attn.cfg.act_fn == "relu"

    assert isinstance(attn.W_K, Params4bit)
    assert isinstance(attn.W_V, Params4bit)
    nq = int((cfg.d_model * cfg.d_model) / 2)
    assert attn.W_K.data.shape == (nq, 1)
    assert attn.W_V.data.shape == (nq, 1)

    assert attn.b_K.shape == (cfg.n_heads, cfg.d_head)
    assert attn.b_V.shape == (cfg.n_heads, cfg.d_head)
    assert torch.all(attn.b_K == 0)
    assert torch.all(attn.b_V == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for half/bfloat16 tests")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_attention_forward_half_precisions(dtype):
    # Construct a small attention block
    cfg = HookedTransformerConfig(
        d_model=64, d_head=16, n_heads=4, n_layers=1, n_ctx=8, dtype=dtype
    )
    attn = Attention(cfg)
    # Random inputs in the matching dtype
    batch = 1
    seq = 4
    x = torch.rand((batch, seq, cfg.d_model), dtype=dtype).to("cuda")
    # Run forward through attention (q,k,v = x)
    out = attn(x, x, x)
    # Should not raise and return a tensor on cuda with same dtype as cfg or compatible
    assert isinstance(out, torch.Tensor)
    assert out.device.type == "cuda"


def test_attention_config_dict():
    cfg = {
        "n_layers": 12,
        "d_model": 512,
        "n_ctx": 1024,
        "d_head": 64,
        "n_heads": 8,
        "load_in_4bit": False,
        "dtype": torch.float32,
        "act_fn": "relu",
    }
    attn = Attention(cfg)
    assert attn.cfg.n_layers == 12
    assert attn.cfg.d_model == 512
    assert attn.cfg.n_ctx == 1024
    assert attn.cfg.d_head == 64
    assert attn.cfg.n_heads == 8
    assert attn.cfg.load_in_4bit == False
    assert attn.cfg.dtype == torch.float32
    assert attn.cfg.act_fn == "relu"


def test_attention_does_not_allocate_full_causal_mask():
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=4,
        n_ctx=8192,
        d_head=2,
        n_heads=2,
        act_fn="relu",
    )

    attn = Attention(cfg)

    assert attn.mask.shape == (0, 0)
    assert attn.state_dict()["mask"].numel() == 0


def test_rotary_embeddings_initial_cache_is_bounded():
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=8,
        n_ctx=8192,
        d_head=4,
        n_heads=2,
        act_fn="relu",
        positional_embedding_type="rotary",
    )

    attn = Attention(cfg)
    rotary_dim = cfg.rotary_dim
    assert rotary_dim is not None

    assert attn.cfg.n_ctx == 8192
    assert attn.rotary_sin.shape == (2048, rotary_dim)
    assert attn.rotary_cos.shape == (2048, rotary_dim)


def test_rotary_embeddings_initial_cache_matches_short_context():
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=8,
        n_ctx=128,
        d_head=4,
        n_heads=2,
        act_fn="relu",
        positional_embedding_type="rotary",
    )

    attn = Attention(cfg)
    rotary_dim = cfg.rotary_dim
    assert rotary_dim is not None

    assert attn.rotary_sin.shape == (cfg.n_ctx, rotary_dim)
    assert attn.rotary_cos.shape == (cfg.n_ctx, rotary_dim)


def test_apply_rotary_extends_embeddings_on_demand():
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=8,
        n_ctx=4096,
        d_head=4,
        n_heads=2,
        act_fn="relu",
        positional_embedding_type="rotary",
    )
    attn = Attention(cfg)
    rotary_dim = cfg.rotary_dim
    assert rotary_dim is not None
    x = torch.randn((1, 2, cfg.n_heads, cfg.d_head), dtype=cfg.dtype)

    out = attn.apply_rotary(x, past_kv_pos_offset=2047)

    expected_sin, expected_cos = attn.calculate_sin_cos_rotary(
        rotary_dim,
        2049,
        base=cfg.rotary_base,
        dtype=cfg.dtype,
    )
    assert out.shape == x.shape
    assert attn.rotary_sin.shape == (2049, rotary_dim)
    assert attn.rotary_cos.shape == (2049, rotary_dim)
    torch.testing.assert_close(attn.rotary_sin, expected_sin)
    torch.testing.assert_close(attn.rotary_cos, expected_cos)


def test_local_rotary_extension_uses_local_base():
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=8,
        n_ctx=4096,
        d_head=4,
        n_heads=2,
        act_fn="relu",
        positional_embedding_type="rotary",
        rotary_base=1_000_000,
        rotary_base_local=10_000,
        window_size=128,
    )
    attn = Attention(cfg, attn_type="local")
    rotary_dim = cfg.rotary_dim
    rotary_base_local = cfg.rotary_base_local
    assert rotary_dim is not None
    assert rotary_base_local is not None

    attn._extend_rotary_embeddings(2050)

    expected_local_sin, expected_local_cos = attn.calculate_sin_cos_rotary(
        rotary_dim,
        2050,
        base=rotary_base_local,
        dtype=cfg.dtype,
    )
    global_sin, _ = attn.calculate_sin_cos_rotary(
        rotary_dim,
        2050,
        base=cfg.rotary_base,
        dtype=cfg.dtype,
    )
    torch.testing.assert_close(attn.rotary_sin, expected_local_sin)
    torch.testing.assert_close(attn.rotary_cos, expected_local_cos)
    assert not torch.allclose(attn.rotary_sin, global_sin)


def test_attention_loads_legacy_full_rotary_buffers_with_strict_true():
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=8,
        n_ctx=4096,
        d_head=4,
        n_heads=2,
        act_fn="relu",
        positional_embedding_type="rotary",
    )
    attn = Attention(cfg)
    rotary_dim = cfg.rotary_dim
    assert rotary_dim is not None
    state_dict = attn.state_dict()
    state_dict["rotary_sin"], state_dict["rotary_cos"] = attn.calculate_sin_cos_rotary(
        rotary_dim,
        cfg.n_ctx,
        base=cfg.rotary_base,
        dtype=cfg.dtype,
    )

    incompatible_keys = attn.load_state_dict(state_dict, strict=True)

    assert incompatible_keys.missing_keys == []
    assert incompatible_keys.unexpected_keys == []
    assert attn.rotary_sin.shape == (2048, rotary_dim)
    assert attn.rotary_cos.shape == (2048, rotary_dim)


def test_apply_causal_mask_global_matches_absolute_positions():
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=4,
        n_ctx=16,
        d_head=2,
        n_heads=2,
        act_fn="relu",
    )
    attn = Attention(cfg)
    attn_scores = torch.zeros((1, 1, 3, 5))

    masked_scores = attn.apply_causal_mask(attn_scores, past_kv_pos_offset=2)

    expected_allowed = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )
    assert torch.equal(torch.isfinite(masked_scores[0, 0]), expected_allowed)


def test_apply_causal_mask_local_matches_window():
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=4,
        n_ctx=16,
        d_head=2,
        n_heads=2,
        act_fn="relu",
        window_size=2,
    )
    attn = Attention(cfg, attn_type="local")
    attn_scores = torch.zeros((1, 1, 3, 5))

    masked_scores = attn.apply_causal_mask(attn_scores, past_kv_pos_offset=2)

    expected_allowed = torch.tensor(
        [
            [False, True, True, False, False],
            [False, False, True, True, False],
            [False, False, False, True, True],
        ]
    )
    assert torch.equal(torch.isfinite(masked_scores[0, 0]), expected_allowed)


def test_apply_causal_mask_combines_padding_mask():
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=4,
        n_ctx=16,
        d_head=2,
        n_heads=2,
        act_fn="relu",
    )
    attn = Attention(cfg)
    attn_scores = torch.zeros((1, 1, 3, 5))
    attention_mask = torch.tensor([[1, 1, 0, 1, 1]])

    masked_scores = attn.apply_causal_mask(
        attn_scores,
        past_kv_pos_offset=2,
        attention_mask=attention_mask,
    )

    expected_allowed = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, False, True, False],
            [True, True, False, True, True],
        ]
    )
    assert torch.equal(torch.isfinite(masked_scores[0, 0]), expected_allowed)


def test_attention_loads_legacy_full_mask_with_strict_true():
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=4,
        n_ctx=16,
        d_head=2,
        n_heads=2,
        act_fn="relu",
    )
    attn = Attention(cfg)
    state_dict = attn.state_dict()
    state_dict["mask"] = torch.ones((cfg.n_ctx, cfg.n_ctx), dtype=torch.bool)

    incompatible_keys = attn.load_state_dict(state_dict, strict=True)

    assert incompatible_keys.missing_keys == []
    assert incompatible_keys.unexpected_keys == []
    assert attn.mask.shape == (0, 0)


def test_remove_einsum_from_complex_attn_linear():
    batch = 64
    pos = 128
    head_index = 8
    d_model = 512
    d_head = 64
    input = torch.randn(batch, pos, head_index, d_model)
    w = torch.randn(head_index, d_model, d_head)
    b = torch.randn(head_index, d_head)
    result_new = complex_attn_linear(input, w, b)

    # Check if new implementation without einsum produces correct shape
    assert result_new.shape == (batch, pos, head_index, d_head)

    # Old implementation used einsum
    result_old = (
        einops.einsum(
            input,
            w,
            "batch pos head_index d_model, head_index d_model d_head -> batch pos head_index d_head",
        )
        + b
    )

    # Check if the results are the same
    assert torch.allclose(result_new, result_old, atol=1e-4)


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or torch.__version__ != "2.8.0",
    reason="Issue with F.linear issue exclusive to mps and PyTorch 2.8\n"
    "https://github.com/pytorch/pytorch/issues/161640",
)
def test_cpu_mps_outputs_match():
    torch.manual_seed(0)

    cfg = {
        "n_layers": 1,
        "d_model": 48,
        "n_ctx": 256,
        "d_head": 16,
        "n_heads": 3,
        "load_in_4bit": False,
        "dtype": torch.float32,
        "act_fn": "relu",
    }

    def init_weights(attn_layer: nn.Module):
        nn.init.normal_(attn_layer.W_Q, mean=0.0, std=0.02)
        nn.init.normal_(attn_layer.W_K, mean=0.0, std=0.02)
        nn.init.normal_(attn_layer.W_V, mean=0.0, std=0.02)
        nn.init.normal_(attn_layer.W_O, mean=0.0, std=0.02)
        return attn_layer

    attn_cpu = Attention(cfg)
    attn_cpu = init_weights(attn_cpu)

    attn_mps = Attention(cfg).to("mps")
    attn_mps.load_state_dict(attn_cpu.state_dict(), strict=True)

    batch = 1
    input_cpu = torch.randn(batch, cfg["n_ctx"], cfg["d_model"])
    input_mps = input_cpu.to("mps")

    cpu_output = attn_cpu(input_cpu, input_cpu, input_cpu)
    mps_output = attn_mps(input_mps, input_mps, input_mps)

    assert torch.allclose(cpu_output, mps_output.cpu())
