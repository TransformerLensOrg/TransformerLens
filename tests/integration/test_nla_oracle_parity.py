"""Deterministic local and slow released-Qwen NLA reference comparisons.

The direct-HF AR path follows NLACritic in kitft/natural_language_autoencoders
at 0577769b55ad4fdd96d159e983361b97fa4e7331 (Apache-2.0): replace final
normalization, execute the truncated backbone, and apply the separate value head.
Oracle code is independently written; no downloaded Python client is executed.
Exact fp32 equality is the initial acceptance criterion, not a calibrated drift
allowance. Any nonzero drift requires investigation and recorded measurements
before introducing a justified tolerance. Released tests are deliberately slow
and run on CPU, one model at a time, without skips for unavailable resources.
"""

import gc
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from huggingface_hub import HfApi
from safetensors.torch import load_file
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis._nla_artifacts import (
    ArtifactRef,
    NLAPairConfig,
    acquire_snapshot,
    load_local_metadata,
    load_snapshot_tokenizer,
    validate_value_head_file,
)
from transformer_lens.tools.analysis._nla_contracts import NLASamplingConfig
from transformer_lens.tools.analysis._nla_reconstructor import (
    ReconstructorRuntime,
    load_critic_model,
    prepare_critic_tokens,
    reconstruct_vector,
    validate_loading_info,
)
from transformer_lens.tools.analysis._nla_target import extract_activation, load_target
from transformer_lens.tools.analysis._nla_verbalizer import (
    VerbalizerRuntime,
    embedding_overwrite,
)
from transformer_lens.utilities.hf_utils import CACHE_DIR, call_hf_with_retry

QWEN_PAIR = NLAPairConfig(
    ArtifactRef("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    ArtifactRef("kitft/nla-qwen2.5-7b-L20-av", "b88469162777ae6553bc14208eb0cb579336f8f4"),
    ArtifactRef("kitft/nla-qwen2.5-7b-L20-ar", "e2c9e57eac213d37a31612087f645ab6332c1bb6"),
    "qwen2",
    3584,
    20,
    28,
)
_FIXED_TEXTS = (
    "The model is considering whether its answer will be evaluated.",
    "A description of ordinary objects and their spatial relationships.",
)


def assert_tensor_parity(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    """Report finite tensor errors and require deterministic exact equivalence."""
    assert actual.shape == expected.shape, label
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), label
    errors = (actual.double() - expected.double()).abs()
    print(f"{label}: max_abs={errors.max().item():.9g}, mean_abs={errors.mean().item():.9g}")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=label)


def actor_reference(
    model: Any, tokens: torch.Tensor, vector: torch.Tensor, position: int, scale: float
) -> torch.Tensor:
    """Construct post-embedding inputs explicitly, independently of the TL hook."""
    source = vector.detach().cpu().double()
    source = source / source.abs().max()
    source = source / torch.linalg.vector_norm(source) * scale
    with torch.inference_mode():
        embeddings = model.get_input_embeddings()(tokens).clone()
        embeddings[0, position] = source.to(embeddings)
        return model(inputs_embeds=embeddings, use_cache=False).logits.detach().cpu().clone()


def critic_reference(model: Any, head: torch.nn.Linear, tokens: torch.Tensor) -> torch.Tensor:
    """Direct-HF reference: final norm bypass plus raw final-token value head."""
    with torch.inference_mode():
        residual = model.model(tokens, use_cache=False).last_hidden_state[0, -1]
        return head(residual).detach().cpu().clone()


def require_storage(refs: tuple[ArtifactRef, ...]) -> None:
    """Fail before large downloads if missing artifact bytes exceed free cache space."""
    required = 0
    for ref in refs:
        info = call_hf_with_retry(
            HfApi().model_info, ref.repo_id, revision=ref.revision, files_metadata=True
        )
        assert info.sha == ref.revision
        snapshot = (
            Path(CACHE_DIR)
            / ("models--" + ref.repo_id.replace("/", "--"))
            / "snapshots"
            / ref.revision
        )
        for file in info.siblings or ():
            if file.rfilename.endswith(".safetensors"):
                assert file.size is not None, "Artifact size must be known before download"
                cached = snapshot / file.rfilename
                if not cached.is_file() or cached.stat().st_size != file.size:
                    required += file.size
    cache = Path(CACHE_DIR)
    ancestor = cache
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    reserve = 2 * 1024**3
    if required + reserve > free:
        raise RuntimeError(
            f"Insufficient cache storage: {required / 1024 ** 3:.1f} GiB of missing weights "
            f"plus 2 GiB reserve required, {free / 1024 ** 3:.1f} GiB free. "
            "Use HF_HOME on a larger volume and rerun the slow tests."
        )


def _normal_activation() -> torch.Tensor:
    target = load_target(QWEN_PAIR, device="cpu")
    try:
        result = extract_activation(
            target,
            QWEN_PAIR,
            "qwen-parity",
            "The auditor asked about the model's internal reasoning.",
            position=-1,
            prepend_bos=False,
        )
        assert result.provenance.source_verified
        return result.vector.clone()
    finally:
        del target
        gc.collect()


def test_parity_assertion_detects_drift():
    with pytest.raises(AssertionError):
        assert_tensor_parity(torch.tensor([1.001]), torch.tensor([1.0]), "deliberate drift")
    with pytest.raises(AssertionError):
        assert_tensor_parity(torch.tensor([float("nan")]), torch.tensor([1.0]), "nonfinite")


def test_direct_hf_oracle_mechanics(tmp_path):
    """Normal-CI sanity check of oracle construction on a tiny local Qwen model."""
    config = Qwen2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    original = Qwen2ForCausalLM(config).eval()
    original.config._attn_implementation = "eager"
    original.save_pretrained(tmp_path)
    original.config.dtype = torch.float32
    tokens = torch.tensor([[1, 2, 3]])
    vector = torch.arange(1.0, 33.0)
    expected = actor_reference(original, tokens, vector, 1, 5.0)
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"UNK": 0}, unk_token="UNK")), unk_token="UNK"
    )
    bridge = TransformerBridge.boot_transformers(
        "local-qwen", hf_model=original, tokenizer=tokenizer, device="cpu"
    )
    bridge.eval()
    with torch.inference_mode(), bridge.hooks(
        fwd_hooks=[("hook_embed", embedding_overwrite(vector, 5.0, 1, 3, cached=True))]
    ):
        actual = bridge(tokens, use_cache=False).cpu()
    assert_tensor_parity(actual, expected, "tiny AV logits")
    pair = NLAPairConfig(
        ArtifactRef("example/target", "a" * 40),
        ArtifactRef("example/av", "b" * 40),
        ArtifactRef("example/ar", "c" * 40),
        "qwen2",
        32,
        1,
        3,
    )
    from transformer_lens.tools.analysis._nla_artifacts import NLAMetadata

    metadata = NLAMetadata(
        "ar",
        32,
        1,
        "marker",
        2,
        "{injection_char}",
        "{explanation}",
        32**0.5,
        None,
        None,
        None,
        (3,),
    )
    original.model.norm = torch.nn.Identity()
    head = torch.nn.Linear(32, 32, bias=False).eval()
    expected_vector = critic_reference(original, head, tokens)
    actual_vector = reconstruct_vector(bridge, head, tokens, metadata, pair).vector
    assert_tensor_parity(actual_vector, expected_vector, "tiny AR vector")


@pytest.mark.slow
def test_released_qwen_actor_prefill_parity():
    require_storage((QWEN_PAIR.target, QWEN_PAIR.av))
    normal = _normal_activation()
    vectors = (normal, torch.linspace(-1.0, 1.0, QWEN_PAIR.d_model))
    directory = acquire_snapshot(QWEN_PAIR.av, include_weights=True)
    tokenizer = load_snapshot_tokenizer(directory)
    metadata = load_local_metadata(directory, QWEN_PAIR, "av", tokenizer)
    assert metadata.actor_token_ids is not None and metadata.injection_position is not None
    assert metadata.injection_scale is not None
    tokens = torch.tensor([metadata.actor_token_ids], dtype=torch.int64)
    direct: Any = AutoModelForCausalLM.from_pretrained(
        str(directory),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    direct.eval()
    expected = [
        actor_reference(direct, tokens, v, metadata.injection_position, metadata.injection_scale)
        for v in vectors
    ]
    del direct
    gc.collect()
    runtime = VerbalizerRuntime(QWEN_PAIR, "qwen-parity", device="cpu", local_files_only=True)
    model = runtime._load(directory, tokenizer)
    try:
        for index, (vector, reference) in enumerate(zip(vectors, expected)):
            hook = embedding_overwrite(
                vector,
                metadata.injection_scale,
                metadata.injection_position,
                tokens.shape[1],
                cached=True,
            )
            with torch.inference_mode(), model.hooks(fwd_hooks=[("hook_embed", hook)]):
                actual = model(tokens, use_cache=False).detach().cpu()
            assert_tensor_parity(actual, reference, f"released AV input {index}")
    finally:
        del model, runtime
        gc.collect()


@pytest.mark.slow
def test_released_qwen_critic_vector_parity():
    require_storage((QWEN_PAIR.ar,))
    directory = acquire_snapshot(QWEN_PAIR.ar, include_weights=True)
    tokenizer = load_snapshot_tokenizer(directory)
    metadata = load_local_metadata(directory, QWEN_PAIR, "ar", tokenizer)
    config = json.loads((directory / "config.json").read_text())
    head_path = directory / "value_head.safetensors"
    validate_value_head_file(head_path, QWEN_PAIR.d_model)
    head = torch.nn.Linear(QWEN_PAIR.d_model, QWEN_PAIR.d_model, bias=False)
    head.load_state_dict(load_file(str(head_path)), strict=True)
    head.eval()
    tokens_list = []
    for text in _FIXED_TEXTS:
        tokens = prepare_critic_tokens(tokenizer, metadata, text, config["max_position_embeddings"])
        independent = tokenizer(
            metadata.critic_template.format(explanation=text),
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"]
        assert torch.equal(tokens, independent)
        tokens_list.append(tokens)
    loaded: Any = AutoModelForCausalLM.from_pretrained(
        str(directory),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    direct, info = loaded
    del loaded
    validate_loading_info(info)
    direct.model.norm = torch.nn.Identity()
    direct.lm_head = torch.nn.Identity()
    direct.eval()
    expected = [critic_reference(direct, head, tokens) for tokens in tokens_list]
    del direct
    gc.collect()
    model = load_critic_model(directory, QWEN_PAIR, tokenizer, "cpu")
    try:
        for index, (tokens, reference) in enumerate(zip(tokens_list, expected)):
            actual = reconstruct_vector(model, head, tokens, metadata, QWEN_PAIR).vector
            assert_tensor_parity(actual, reference, f"released AR text {index}")
    finally:
        del model
        gc.collect()


@pytest.mark.slow
def test_released_qwen_sequential_round_trip():
    require_storage((QWEN_PAIR.target, QWEN_PAIR.av, QWEN_PAIR.ar))
    activation = _normal_activation()
    actor = VerbalizerRuntime(QWEN_PAIR, "qwen-parity", device="cpu")
    try:
        explanation = actor.verbalize(activation, NLASamplingConfig(max_new_tokens=512))
    finally:
        actor.unload()
        del actor
        gc.collect()
    assert (
        explanation.text is not None
    ), f"AV parse failure: {explanation.parse_status}; {explanation.raw_text!r}"
    critic = ReconstructorRuntime(QWEN_PAIR, device="cpu")
    try:
        result = critic.reconstruct(explanation.text)
    finally:
        critic.unload()
    assert result.vector.shape == activation.shape
    assert torch.isfinite(result.vector).all()
    assert torch.any(result.vector != 0)
