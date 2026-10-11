"""Local artifact validation without model or Hub loading."""

import copy
import json
import math

import pytest
import torch
import yaml
from safetensors.torch import save_file

from transformer_lens.tools.analysis._nla_artifacts import (
    ArtifactRef,
    NLAPairConfig,
    load_local_metadata,
    load_registry,
    parse_registry,
    validate_metadata,
    validate_snapshot_path,
    validate_value_head,
    validate_value_head_file,
)


class LocalTokenizer:
    unk_token_id = 0

    def __init__(self):
        self.injection_ids = [7]
        self.actor_ids = [1, 2, 7, 3, 4]
        self.critic_ids = [1, 8, 9]

    def encode(self, text, *, add_special_tokens):
        if text == "MARK":
            assert add_special_tokens is False
            return self.injection_ids
        assert text == "Summary: x END"
        assert add_special_tokens is True
        return self.critic_ids

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert conversation == [{"role": "user", "content": "Explain MARK now"}]
        assert tokenize is True and add_generation_prompt is True
        return self.actor_ids


@pytest.fixture
def pair():
    return NLAPairConfig(
        ArtifactRef("example/target", "a" * 40),
        ArtifactRef("example/av", "b" * 40),
        ArtifactRef("example/ar", "c" * 40),
        "qwen2",
        4,
        1,
        3,
    )


@pytest.fixture
def actor():
    return {
        "kind": "nla_model",
        "schema_version": 2,
        "role": "av",
        "d_model": 4,
        "extraction_layer_index": 1,
        "extraction": {"injection_scale": 150.0, "mse_scale": 2.0},
        "tokens": {
            "injection_char": "MARK",
            "injection_token_id": 7,
            "injection_left_neighbor_id": 2,
            "injection_right_neighbor_id": 3,
            "critic_suffix_ids": None,
        },
        "prompt_templates": {
            "av": "Explain {injection_char} now",
            "ar": "Summary: {explanation} END",
        },
    }


@pytest.fixture
def critic(actor):
    result = copy.deepcopy(actor)
    result["role"] = "ar"
    result["extraction"]["injection_scale"] = None
    result["tokens"]["critic_suffix_ids"] = [8, 9]
    result["critic"] = {"extraction_layer_index": 1}
    return result


def config(role):
    return {"model_type": "qwen2", "hidden_size": 4, "num_hidden_layers": 3 if role == "av" else 2}


def test_valid_roles(actor, critic, pair):
    av = validate_metadata(actor, config("av"), pair, "av", LocalTokenizer())
    ar = validate_metadata(critic, config("ar"), pair, "ar", LocalTokenizer())
    assert av.actor_token_ids == (1, 2, 7, 3, 4)
    assert av.injection_position == 2 and av.injection_scale == 150
    assert av.critic_suffix_ids is None
    assert ar.critic_suffix_ids == (8, 9)
    assert ar.actor_token_ids is None and ar.injection_scale is None


@pytest.mark.parametrize(
    "path,value",
    [
        (("kind",), "nla_dataset"),
        (("schema_version",), 1),
        (("schema_version",), True),
        (("role",), "ar"),
        (("d_model",), 5),
        (("d_model",), False),
        (("d_model",), 0),
        (("extraction_layer_index",), 0),
        (("extraction_layer_index",), True),
        (("extraction", "injection_scale"), None),
        (("extraction", "injection_scale"), 0),
        (("extraction", "injection_scale"), -1),
        (("extraction", "injection_scale"), float("inf")),
        (("extraction", "injection_scale"), True),
        (("extraction", "mse_scale"), None),
        (("extraction", "mse_scale"), float("nan")),
        (("extraction", "mse_scale"), 1.99),
        (("tokens", "injection_token_id"), True),
        (("tokens", "injection_left_neighbor_id"), 10),
        (("tokens", "injection_right_neighbor_id"), 10),
        (("tokens", "injection_char"), ""),
        (("tokens", "critic_suffix_ids"), [8]),
        (("prompt_templates", "av"), "{unknown}"),
        (("prompt_templates", "av"), "{injection_char}{injection_char}"),
        (("prompt_templates", "av"), "{injection_char!r}"),
        (("prompt_templates", "ar"), "{explanation:>10}"),
        (("prompt_templates", "ar"), "{"),
        (("tokens",), []),
        (("extraction",), None),
    ],
)
def test_invalid_actor_metadata(actor, pair, path, value):
    mapping = actor
    for key in path[:-1]:
        mapping = mapping[key]
    mapping[path[-1]] = value
    with pytest.raises(ValueError):
        validate_metadata(actor, config("av"), pair, "av", LocalTokenizer())


@pytest.mark.parametrize(
    "key",
    [
        "kind",
        "schema_version",
        "role",
        "d_model",
        "tokens",
        "extraction",
        "prompt_templates",
        "extraction_layer_index",
    ],
)
def test_missing_metadata_fields(actor, pair, key):
    actor.pop(key)
    with pytest.raises(ValueError):
        validate_metadata(actor, config("av"), pair, "av", LocalTokenizer())


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_type", "llama"),
        ("hidden_size", 8),
        ("hidden_size", True),
        ("num_hidden_layers", 2),
        ("num_hidden_layers", False),
    ],
)
def test_invalid_model_config(actor, pair, field, value):
    model = config("av")
    model[field] = value
    with pytest.raises(ValueError):
        validate_metadata(actor, model, pair, "av", LocalTokenizer())


@pytest.mark.parametrize("ids", [[], [7, 7], [0], [8], [True]])
def test_reserved_token_drift(actor, pair, ids):
    tokenizer = LocalTokenizer()
    tokenizer.injection_ids = ids
    with pytest.raises(ValueError):
        validate_metadata(actor, config("av"), pair, "av", tokenizer)


def test_unknown_reserved_token(actor, pair):
    tokenizer = LocalTokenizer()
    tokenizer.unk_token_id = 7
    with pytest.raises(ValueError, match="UNK"):
        validate_metadata(actor, config("av"), pair, "av", tokenizer)


@pytest.mark.parametrize("ids", [[1, 2, 3], [2, 7, 7, 3], [7, 3], [2, 7], [2, 7, True]])
def test_placeholder_occurrence_and_position(actor, pair, ids):
    tokenizer = LocalTokenizer()
    tokenizer.actor_ids = ids
    with pytest.raises(ValueError):
        validate_metadata(actor, config("av"), pair, "av", tokenizer)


@pytest.mark.parametrize(
    "path,value",
    [
        (("tokens", "critic_suffix_ids"), None),
        (("tokens", "critic_suffix_ids"), []),
        (("tokens", "critic_suffix_ids"), [9, 8]),
        (("tokens", "critic_suffix_ids"), [True]),
        (("extraction", "injection_scale"), 150),
        (("critic", "extraction_layer_index"), 0),
    ],
)
def test_invalid_critic_metadata(critic, pair, path, value):
    critic[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        validate_metadata(critic, config("ar"), pair, "ar", LocalTokenizer())


def test_wrong_critic_depth(critic, pair):
    with pytest.raises(ValueError, match="depth"):
        validate_metadata(critic, config("av"), pair, "ar", LocalTokenizer())


def test_qwen_scale_serialization(actor, pair):
    large = NLAPairConfig(pair.target, pair.av, pair.ar, "qwen2", 3584, 20, 28)
    actor["d_model"] = 3584
    actor["extraction_layer_index"] = 20
    actor["extraction"]["mse_scale"] = 59.86651818838306
    av_config = {"model_type": "qwen2", "hidden_size": 3584, "num_hidden_layers": 28}
    assert math.isclose(
        validate_metadata(actor, av_config, large, "av", LocalTokenizer()).mse_scale,
        math.sqrt(3584),
    )
    actor["extraction"]["mse_scale"] += 1e-8
    with pytest.raises(ValueError, match="mse_scale"):
        validate_metadata(actor, av_config, large, "av", LocalTokenizer())


def test_local_metadata_files(tmp_path, actor, pair):
    (tmp_path / "nla_meta.yaml").write_text(yaml.safe_dump(actor))
    (tmp_path / "config.json").write_text(json.dumps(config("av")))
    assert load_local_metadata(tmp_path, pair, "av", LocalTokenizer()).d_model == 4
    (tmp_path / "nla_meta.yaml").write_text("!!python/object:builtins.object {}")
    with pytest.raises(ValueError, match="syntax"):
        load_local_metadata(tmp_path, pair, "av", LocalTokenizer())


@pytest.mark.parametrize("revision", ["main", "a" * 39, "A" * 40, "../snapshot", ""])
def test_immutable_revisions(revision):
    with pytest.raises(ValueError, match="revision"):
        ArtifactRef("example/av", revision)


@pytest.mark.parametrize(
    "repo", ["../av", "example/../av", "av", "/example/av", "./av", "example/..", "example/."]
)
def test_repo_identifiers(repo):
    with pytest.raises(ValueError, match="repo_id"):
        ArtifactRef(repo, "a" * 40)


def registry_entry():
    return {
        "target": {"repo_id": "example/target", "revision": "a" * 40},
        "av": {"repo_id": "example/av", "revision": "b" * 40},
        "ar": {"repo_id": "example/ar", "revision": "c" * 40},
        "model_type": "qwen2",
        "d_model": 4,
        "layer": 1,
        "target_layers": 3,
    }


def test_registry_resource_and_parser(pair):
    assert load_registry() == {}
    assert parse_registry({"schema_version": 1, "models": {"tiny": registry_entry()}}) == {
        "tiny": pair
    }


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"schema_version": True, "models": {}},
        {"schema_version": 2, "models": {}},
        {"schema_version": 1, "models": []},
    ],
)
def test_invalid_registry(value):
    with pytest.raises(ValueError):
        parse_registry(value)


@pytest.mark.parametrize(
    "key,value", [("d_model", True), ("layer", -1), ("layer", 3), ("target_layers", 0)]
)
def test_invalid_pair_contract(key, value):
    entry = registry_entry()
    entry[key] = value
    with pytest.raises(ValueError):
        parse_registry({"schema_version": 1, "models": {"tiny": entry}})


def test_distinct_artifact_identities():
    entry = registry_entry()
    entry["ar"] = entry["av"]
    with pytest.raises(ValueError, match="distinct"):
        parse_registry({"schema_version": 1, "models": {"tiny": entry}})


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"weight": torch.eye(4), "bias": torch.zeros(4)},
        {"weight": torch.eye(4), "other": torch.zeros(1)},
        {"weight": torch.eye(3)},
        {"weight": torch.zeros(4, 4, dtype=torch.int64)},
        {"weight": torch.full((4, 4), float("nan"))},
    ],
)
def test_invalid_value_heads(state):
    with pytest.raises(ValueError):
        validate_value_head(state, 4)


def test_snapshot_path_identity(tmp_path, pair):
    path = tmp_path / "models--example--av" / "snapshots" / pair.av.revision
    assert validate_snapshot_path(path, pair.av) == path
    for invalid in (
        path.with_name(pair.ar.revision),
        tmp_path / "models--example--ar" / "snapshots" / pair.av.revision,
        tmp_path / pair.av.revision,
    ):
        with pytest.raises(ValueError, match="repository and revision"):
            validate_snapshot_path(invalid, pair.av)


def test_chat_template_batch_encoding(actor, pair):
    from transformers import BatchEncoding

    tokenizer = LocalTokenizer()
    tokenizer.actor_ids = BatchEncoding({"input_ids": [1, 2, 7, 3, 4]})
    result = validate_metadata(actor, config("av"), pair, "av", tokenizer)
    assert result.injection_position == 2


@pytest.mark.parametrize(
    "role,section,key",
    [
        ("av", "tokens", "critic_suffix_ids"),
        ("ar", "extraction", "injection_scale"),
    ],
)
def test_explicit_nullable_fields(actor, critic, pair, role, section, key):
    meta = actor if role == "av" else critic
    del meta[section][key]
    with pytest.raises(ValueError, match="must record"):
        validate_metadata(meta, config(role), pair, role, LocalTokenizer())


def test_value_head_safetensors(tmp_path):
    path = tmp_path / "value_head.safetensors"
    save_file({"weight": torch.eye(4)}, path)
    validate_value_head_file(path, 4)
    save_file({"weight": torch.eye(3)}, path)
    with pytest.raises(ValueError, match="shape"):
        validate_value_head_file(path, 4)
    save_file({"weight": torch.eye(4), "bias": torch.zeros(4)}, path)
    with pytest.raises(ValueError, match="bias"):
        validate_value_head_file(path, 4)
