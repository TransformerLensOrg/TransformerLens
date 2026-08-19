import einops
import pytest
import torch
import torch.nn as nn
from transformers.utils import is_bitsandbytes_available

from transformer_lens.components import Attention
from transformer_lens.components import abstract_attention as abstract_attention_module
from transformer_lens.config import HookedTransformerConfig
from transformer_lens.utilities.attention import complex_attn_linear, simple_attn_linear

if is_bitsandbytes_available():
    from bitsandbytes.nn.modules import Params4bit


class FakeParams4bit:
    def __init__(self, dequantized: torch.Tensor):
        self.data = dequantized
        self.quant_state = object()

    def t(self):
        return self.data.t()


class FakeBnbFunctional:
    @staticmethod
    def dequantize_4bit(input, quant_state):
        return input


class FakeBnb:
    functional = FakeBnbFunctional()

    @staticmethod
    def matmul_4bit(input, weight, bias, quant_state):
        if input.ndim != 3:
            raise AssertionError("split QKV projection should dequantize once")
        return torch.matmul(input, weight)


def fake_4bit_weight(weight: torch.Tensor) -> FakeParams4bit:
    dequantized = einops.rearrange(weight, "head d_model d_head -> (head d_head) d_model")
    return FakeParams4bit(dequantized)


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


@pytest.mark.parametrize("use_split_qkv_input", [False, True])
def test_attention_4bit_qkv_projection_matches_unquantized(monkeypatch, use_split_qkv_input):
    monkeypatch.setattr(abstract_attention_module, "Params4bit", FakeParams4bit, raising=False)
    monkeypatch.setattr(abstract_attention_module, "bnb", FakeBnb, raising=False)

    torch.manual_seed(0)
    cfg = HookedTransformerConfig(
        n_layers=1,
        d_model=6,
        n_ctx=4,
        d_head=2,
        n_heads=3,
        load_in_4bit=False,
        use_split_qkv_input=use_split_qkv_input,
        dtype=torch.float32,
        act_fn="relu",
    )
    attn = Attention(cfg)
    attn.cfg.load_in_4bit = True

    W_Q = torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head)
    W_K = torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head)
    W_V = torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head)

    for name, weight in (("W_Q", W_Q), ("W_K", W_K), ("W_V", W_V)):
        del attn._parameters[name]
        setattr(attn, name, fake_4bit_weight(weight))

    with torch.no_grad():
        attn.b_Q.copy_(torch.randn_like(attn.b_Q))
        attn.b_K.copy_(torch.randn_like(attn.b_K))
        attn.b_V.copy_(torch.randn_like(attn.b_V))

    if use_split_qkv_input:
        query_input = torch.randn(2, 4, cfg.n_heads, cfg.d_model)
        key_input = torch.randn(2, 4, cfg.n_heads, cfg.d_model)
        value_input = torch.randn(2, 4, cfg.n_heads, cfg.d_model)
        expected_fn = complex_attn_linear
    else:
        query_input = torch.randn(2, 4, cfg.d_model)
        key_input = torch.randn(2, 4, cfg.d_model)
        value_input = torch.randn(2, 4, cfg.d_model)
        expected_fn = simple_attn_linear

    q, k, v = attn.calculate_qkv_matrices(query_input, key_input, value_input)

    assert torch.allclose(q, expected_fn(query_input, W_Q, attn.b_Q))
    assert torch.allclose(k, expected_fn(key_input, W_K, attn.b_K))
    assert torch.allclose(v, expected_fn(value_input, W_V, attn.b_V))


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


def test_attention_clip_qkv_clamps_between_projection_and_scores():
    """cfg.clip_qkv must clamp Q/K/V after projection (OLMo v1 / OLMoE semantics):
    output with clip_qkv equals a clip-free module fed pre-clamped Q/K/V."""
    cfg_kwargs = dict(n_layers=1, d_model=32, n_ctx=8, d_head=8, n_heads=4, act_fn="relu")
    # fork_rng: deterministic setup without mutating the global RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        attn_clip = Attention(HookedTransformerConfig(**cfg_kwargs, clip_qkv=0.5))
        attn_ref = Attention(HookedTransformerConfig(**cfg_kwargs))
        # Standalone components carry torch.empty weights (init_weights runs at
        # model level), so fill them explicitly before sharing the state dict.
        with torch.no_grad():
            for param in attn_clip.parameters():
                param.normal_(0, 0.02)
        attn_ref.load_state_dict(attn_clip.state_dict())

        batch, seq = 1, 4
        q0 = torch.randn(batch, seq, 4, 8) * 2
        k0 = torch.randn(batch, seq, 4, 8) * 2
        v0 = torch.randn(batch, seq, 4, 8) * 2
        x = torch.randn(batch, seq, 32)
    # clamp is active on all three of Q/K/V
    assert (q0.abs() > 0.5).any() and (k0.abs() > 0.5).any() and (v0.abs() > 0.5).any()

    attn_clip.calculate_qkv_matrices = lambda *args, **kwargs: (q0, k0, v0)
    attn_ref.calculate_qkv_matrices = lambda *args, **kwargs: (
        q0.clamp(min=-0.5, max=0.5),
        k0.clamp(min=-0.5, max=0.5),
        v0.clamp(min=-0.5, max=0.5),
    )

    with torch.no_grad():
        out_clip = attn_clip(x, x, x)
        out_ref = attn_ref(x, x, x)

    torch.testing.assert_close(out_clip, out_ref)


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
        cfg.n_ctx,
        base=cfg.rotary_base,
        dtype=cfg.dtype,
    )
    assert out.shape == x.shape
    assert attn.rotary_sin.shape == (cfg.n_ctx, rotary_dim)
    assert attn.rotary_cos.shape == (cfg.n_ctx, rotary_dim)
    torch.testing.assert_close(attn.rotary_sin, expected_sin)
    torch.testing.assert_close(attn.rotary_cos, expected_cos)


def test_apply_rotary_extends_with_headroom_for_token_generation(monkeypatch):
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
    x = torch.randn((1, 1, cfg.n_heads, cfg.d_head), dtype=cfg.dtype)
    extension_sizes = []
    original_extend = attn._extend_rotary_embeddings

    def record_extension(new_size):
        extension_sizes.append(new_size)
        original_extend(new_size)

    monkeypatch.setattr(attn, "_extend_rotary_embeddings", record_extension)

    attn.apply_rotary(x, past_kv_pos_offset=2048)
    attn.apply_rotary(x, past_kv_pos_offset=2049)

    assert extension_sizes == [4096]
    assert attn.rotary_sin.shape == (4096, rotary_dim)
    assert attn.rotary_cos.shape == (4096, rotary_dim)


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


class TestLogNAttentionScaling:
    """Qwen-1's log-n scaling: queries past the training length scale by
    log_{train_len}(position), eval only. TL dropped it entirely."""

    @staticmethod
    def _attention(use_logn):
        from transformer_lens.components import Attention
        from transformer_lens.config import HookedTransformerConfig

        # n_ctx larger than the training length: the only configuration in
        # which contexts long enough to trigger log-n can run at all.
        cfg = HookedTransformerConfig(
            d_model=16,
            d_head=4,
            n_heads=4,
            n_ctx=16,
            n_layers=1,
            d_vocab=32,
            act_fn="relu",
            positional_embedding_type="rotary",
            rotary_dim=4,
            use_logn_attn=use_logn,
            train_seq_length=8,
        )
        attn = Attention(cfg).eval()
        # Component construction leaves weights empty (model-level init fills
        # them); zero weights would make every comparison vacuously equal.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            for parameter in attn.parameters():
                torch.nn.init.normal_(parameter, std=0.2)
        return attn

    def test_matches_the_remote_formula_past_n_ctx(self) -> None:
        import math

        attn = self._attention(True)
        q = torch.ones(1, 12, 4, 4)
        scaled = attn._apply_logn_scaling(q, kv_cache_pos_offset=0)
        # Positions 1..8 (<= the training length) untouched; 9..12 scale by log_8(pos).
        torch.testing.assert_close(scaled[:, :8], q[:, :8])
        for position in (9, 12):
            expected = math.log(position, 8)
            torch.testing.assert_close(
                scaled[0, position - 1, 0, 0], torch.tensor(expected), atol=1e-6, rtol=0
            )

    def test_cached_decode_uses_absolute_positions(self) -> None:
        import math

        attn = self._attention(True)
        q = torch.ones(1, 1, 4, 4)  # single decode step at absolute position 11
        scaled = attn._apply_logn_scaling(q, kv_cache_pos_offset=10)
        torch.testing.assert_close(
            scaled[0, 0, 0, 0], torch.tensor(math.log(11, 8)), atol=1e-6, rtol=0
        )

    def test_flag_changes_the_forward_and_default_does_not(self) -> None:
        """End to end through the component forward, past the training length."""
        torch.manual_seed(1)
        x = torch.randn(1, 12, 16)
        on, off = self._attention(True), self._attention(False)
        off.load_state_dict(on.state_dict())
        with torch.no_grad():
            out_on = on(x, x, x)
            out_off = off(x, x, x)
        # Positions within the training length agree exactly; beyond it they differ.
        torch.testing.assert_close(out_on[:, :8], out_off[:, :8])
        assert not torch.allclose(out_on[:, 8:], out_off[:, 8:])

    def test_training_mode_is_untouched(self) -> None:
        attn = self._attention(True).train()
        x = torch.randn(1, 12, 16)
        reference = self._attention(False).train()
        reference.load_state_dict(attn.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(attn(x, x, x), reference(x, x, x))


class TestDynamicNTKRotary:
    """Qwen-1's dynamic NTK widens the rotary BASE past the training length, so
    TL dropping it left long contexts on training-length frequencies."""

    @staticmethod
    def _attention(use_ntk: bool, n_ctx: int = 64):
        from transformer_lens.components import Attention
        from transformer_lens.config import HookedTransformerConfig

        cfg = HookedTransformerConfig(
            d_model=16,
            d_head=4,
            n_heads=4,
            n_ctx=n_ctx,
            n_layers=1,
            d_vocab=32,
            act_fn="relu",
            positional_embedding_type="rotary",
            rotary_dim=4,
            rotary_base=10000,
            use_dynamic_ntk_rope=use_ntk,
            train_seq_length=8,
        )
        attn = Attention(cfg).eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            for parameter in attn.parameters():
                torch.nn.init.normal_(parameter, std=0.2)
        return attn

    def test_alpha_matches_the_remote_step_function(self) -> None:
        """The hub's get_ntk_alpha: 2**ceil(log2(len/train) + 1) - 1, floored at 1."""
        attn = self._attention(True)
        assert attn._ntk_alpha(4, 8) == 1.0  # below the training length
        assert attn._ntk_alpha(8, 8) == 1.0  # exactly at it
        assert attn._ntk_alpha(9, 8) == 3  # just past -> first step
        assert attn._ntk_alpha(16, 8) == 3
        assert attn._ntk_alpha(17, 8) == 7  # 2x past -> next step
        assert attn._ntk_alpha(32, 8) == 7

    def test_base_widens_by_the_remote_exponent(self) -> None:
        attn = self._attention(True)
        before = attn.rotary_cos.clone()
        attn._rescale_rotary_for_ntk(32)  # alpha = 7, rotary_dim = 4
        expected_base = 10000 * 7 ** (4 / (4 - 2))
        expected_sin, expected_cos = attn.calculate_sin_cos_rotary(
            4, attn.rotary_cos.shape[0], base=expected_base, dtype=attn.cfg.dtype
        )
        torch.testing.assert_close(attn.rotary_cos, expected_cos)
        torch.testing.assert_close(attn.rotary_sin, expected_sin)
        assert not torch.allclose(attn.rotary_cos[: before.shape[0]], before)

    def test_short_context_leaves_the_table_alone(self) -> None:
        attn = self._attention(True)
        before = attn.rotary_cos.clone()
        attn._rescale_rotary_for_ntk(8)  # at the training length: alpha = 1
        torch.testing.assert_close(attn.rotary_cos, before)

    def test_forward_differs_only_past_the_training_length(self) -> None:
        torch.manual_seed(1)
        on, off = self._attention(True), self._attention(False)
        off.load_state_dict(on.state_dict())
        with torch.no_grad():
            short = torch.randn(1, 8, 16)
            torch.testing.assert_close(on(short, short, short), off(short, short, short))
            long = torch.randn(1, 32, 16)
            assert not torch.allclose(on(long, long, long), off(long, long, long))

    def test_training_mode_is_untouched(self) -> None:
        on = self._attention(True).train()
        off = self._attention(False).train()
        off.load_state_dict(on.state_dict())
        x = torch.randn(1, 32, 16)
        with torch.no_grad():
            torch.testing.assert_close(on(x, x, x), off(x, x, x))

    def test_table_reverts_when_the_context_shrinks(self) -> None:
        """alpha is recomputed per forward, so a later short sequence must not
        keep the widened base from an earlier long one."""
        attn = self._attention(True)
        baseline = attn.rotary_cos.clone()
        attn._rescale_rotary_for_ntk(32)
        assert not torch.allclose(attn.rotary_cos[: baseline.shape[0]], baseline)
        attn._rescale_rotary_for_ntk(8)
        torch.testing.assert_close(attn.rotary_cos[: baseline.shape[0]], baseline)
