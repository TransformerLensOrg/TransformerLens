"""Facade execution with real local synthetic Qwen artifacts, without Hub mocks."""

import gc
import json
import weakref

import pytest
import torch
import yaml
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis._nla import NaturalLanguageAutoencoder
from transformer_lens.tools.analysis._nla_artifacts import ArtifactRef, NLAPairConfig
from transformer_lens.tools.analysis._nla_contracts import NLASamplingConfig


@pytest.fixture
def local_artifacts(tmp_path):
    backend = Tokenizer(BPE(unk_token="UNK"))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        ["left MARK right Summary x END <explanation>objects in space</explanation>"] * 10,
        BpeTrainer(
            vocab_size=300, special_tokens=["UNK", "MARK"], initial_alphabet=ByteLevel.alphabet()
        ),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="UNK")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    config = Qwen2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=tokenizer.vocab_size + 1,
        max_position_embeddings=1024,
        tie_word_embeddings=False,
        eos_token_id=None,
        bos_token_id=None,
    )
    pair = NLAPairConfig(
        ArtifactRef("synthetic/target", "a" * 40),
        ArtifactRef("synthetic/av", "b" * 40),
        ArtifactRef("synthetic/ar", "c" * 40),
        "qwen2",
        32,
        1,
        3,
    )
    for role, ref in (("av", pair.av), ("ar", pair.ar)):
        directory = (
            tmp_path / ("models--" + ref.repo_id.replace("/", "--")) / "snapshots" / ref.revision
        )
        directory.mkdir(parents=True)
        role_config = Qwen2Config.from_dict(config.to_dict())
        if role == "ar":
            role_config.num_hidden_layers = 2
            role_config.layer_types = role_config.layer_types[:2]
        model = Qwen2ForCausalLM(role_config).eval()
        role_config.save_pretrained(directory)
        state = model.state_dict()
        if role == "ar":
            state = {
                key: value
                for key, value in state.items()
                if key not in {"model.norm.weight", "lm_head.weight"}
            }
            save_file({"weight": torch.eye(32)}, directory / "value_head.safetensors")
        save_file(state, directory / "model.safetensors")
        tokenizer.save_pretrained(directory)
        tokenizer_config_path = directory / "tokenizer_config.json"
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
        tokenizer_config["tokenizer_class"] = "TokenizersBackend"
        tokenizer_config_path.write_text(json.dumps(tokenizer_config))
        from transformers import AutoTokenizer

        live = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        marker = live.encode("MARK", add_special_tokens=False)[0]
        actor_ids = live.apply_chat_template(
            [{"role": "user", "content": "left MARK right"}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if not isinstance(actor_ids, list):
            actor_ids = actor_ids["input_ids"]
        position = actor_ids.index(marker)
        suffix = live.encode("Summary x END", add_special_tokens=True)[-1:]
        metadata = {
            "kind": "nla_model",
            "schema_version": 2,
            "role": role,
            "d_model": 32,
            "extraction_layer_index": 1,
            "extraction": {
                "injection_scale": 5.0 if role == "av" else None,
                "mse_scale": 32**0.5,
            },
            "tokens": {
                "injection_char": "MARK",
                "injection_token_id": marker,
                "injection_left_neighbor_id": actor_ids[position - 1],
                "injection_right_neighbor_id": actor_ids[position + 1],
                "critic_suffix_ids": suffix if role == "ar" else None,
            },
            "prompt_templates": {
                "av": "left {injection_char} right",
                "ar": "Summary {explanation} END",
            },
        }
        (directory / "nla_meta.yaml").write_text(yaml.safe_dump(metadata))
    target_directory = tmp_path / "models--synthetic--target" / "snapshots" / pair.target.revision
    target_directory.mkdir(parents=True)
    original = Qwen2ForCausalLM(config).eval()
    original.save_pretrained(target_directory)
    tokenizer.save_pretrained(target_directory)
    original.config.dtype = torch.float32
    original.config._name_or_path = pair.target.repo_id
    original.config._commit_hash = pair.target.revision
    target = TransformerBridge.boot_transformers(
        pair.target.repo_id, hf_model=original, tokenizer=tokenizer, device="cpu"
    )
    target.eval()
    return pair, target, tmp_path


def test_facade_local_round_trip_and_reload(local_artifacts):
    pair, target, cache = local_artifacts
    with NaturalLanguageAutoencoder(
        pair, "synthetic", cache_dir=str(cache), local_files_only=True
    ) as nla:
        loaded_target = nla.load_target()
        activation = nla.extract(loaded_target, torch.tensor([1, 2, 3]))
        del loaded_target
        explanation = nla.verbalize(activation, max_new_tokens=2, do_sample=False)
        assert explanation.provenance.source_verified
        assert 0 < len(explanation.token_ids) <= 2
        nla.unload("av")
        assert nla._av._model is None
        again = nla.verbalize(activation, max_new_tokens=2, do_sample=False)
        assert again.token_ids == explanation.token_ids
        nla.unload("av")
        reconstruction = nla.reconstruct("x")
        assert reconstruction.vector.shape == (32,) and not reconstruction.vector.requires_grad
        nla.unload("ar")
        again_vector = nla.reconstruct("x").vector
        torch.testing.assert_close(again_vector, reconstruction.vector)
        assert -1 <= nla.score(activation, reconstruction).cosine <= 1
        combined = nla.explain(
            target,
            torch.tensor([1, 2, 3]),
            sampling=NLASamplingConfig(max_new_tokens=2, do_sample=False),
        )
        assert combined.provenance.position == 2
        with pytest.raises(ValueError, match="wrappers"):
            nla.reconstruct("<explanation>x</explanation>")
    assert nla._closed and nla._av._model is None and nla._ar._head is None


def test_generated_explanation_reaches_reconstructor(local_artifacts):
    pair, _, cache = local_artifacts
    with NaturalLanguageAutoencoder(
        pair, "synthetic", cache_dir=str(cache), local_files_only=True
    ) as nla:
        target = nla.load_target()
        target_ref = weakref.ref(target)
        activation = nla.extract(target, torch.tensor([1, 2, 3]))
        del target
        gc.collect()
        assert target_ref() is None
        nla.verbalize(activation, max_new_tokens=1, do_sample=False)
        actor = nla._av._model
        expected_text = "objects in space"
        continuation = actor.tokenizer.encode(
            f"<explanation>{expected_text}</explanation>", add_special_tokens=False
        )
        step = 0

        def force_continuation(logits, hook):
            nonlocal step
            output = torch.full_like(logits, -1e9)
            output[:, -1, continuation[step]] = 1e9
            step += 1
            return output

        with actor.hooks(fwd_hooks=[("unembed.hook_out", force_continuation)]):
            explanation = nla.verbalize(
                activation, max_new_tokens=len(continuation), do_sample=False
            )
        assert explanation.text == expected_text
        assert explanation.token_ids == tuple(continuation)
        actor_ref = weakref.ref(actor)
        del actor
        nla.unload("av")
        gc.collect()
        assert actor_ref() is None
        assert nla._ar._model is None
        reconstruction = nla.reconstruct(explanation.text)
        head = nla._ar._head
        model = nla._ar._model
        raw_prompt = f"Summary {expected_text} END"
        ids = model.tokenizer.encode(raw_prompt, add_special_tokens=True)
        with torch.no_grad():
            residual = model.original_model.model(
                torch.tensor([ids]), use_cache=False
            ).last_hidden_state[0, -1]
            expected = head(residual)
        torch.testing.assert_close(reconstruction.vector, expected)
        del head, model
        nla.unload("ar")
        assert nla._ar._model is None and nla._ar._head is None


def test_facade_does_not_own_target(local_artifacts):
    pair, target, cache = local_artifacts
    nla = NaturalLanguageAutoencoder(pair, "synthetic", cache_dir=str(cache), local_files_only=True)
    original = Qwen2ForCausalLM(target.original_model.config).eval()
    original.config._name_or_path = pair.target.repo_id
    original.config._commit_hash = pair.target.revision
    owned = TransformerBridge.boot_transformers(
        pair.target.repo_id, hf_model=original, tokenizer=target.tokenizer, device="cpu"
    )
    owned.eval()
    result = nla.extract(owned, torch.tensor([1, 2, 3]))
    ref = weakref.ref(owned)
    del owned, original
    gc.collect()
    assert ref() is None
    assert result.vector.device.type == "cpu"
    nla.close()
