#!/usr/bin/env python3
"""Analyze a HuggingFace model to extract information needed for an Architecture Adapter.

Usage:
    python scripts/analyze-hf-model.py <model_name_or_path>

Example:
    python scripts/analyze-hf-model.py meta-llama/Llama-3.1-8B
    python scripts/analyze-hf-model.py google/gemma-2-2b

Outputs:
    - Architecture class name
    - Config field mappings (HF -> TL)
    - Normalization type
    - Positional embedding type
    - Attention type (MHA/GQA/MQA)
    - MLP type (gated/standard)
    - Module path listing
    - Suggested closest existing adapter
"""

import argparse
import json
import re
import sys
from pathlib import Path


def analyze_config(config) -> dict:
    """Extract adapter-relevant fields from a HuggingFace config."""
    info = {}

    # Architecture class
    info["architecture_class"] = getattr(config, "architectures", ["Unknown"])[0]
    info["model_type"] = getattr(config, "model_type", "unknown")

    # Dimensions
    info["d_model"] = getattr(config, "hidden_size", None)
    info["n_heads"] = getattr(config, "num_attention_heads", None)
    info["n_key_value_heads"] = getattr(config, "num_key_value_heads", None)
    info["d_mlp"] = getattr(config, "intermediate_size", None)
    info["n_layers"] = getattr(config, "num_hidden_layers", None)
    info["d_vocab"] = getattr(config, "vocab_size", None)
    info["n_ctx"] = getattr(config, "max_position_embeddings", None)

    # Derived — guard against None or zero
    if info["d_model"] and info["n_heads"] and info["n_heads"] > 0:
        info["d_head"] = info["d_model"] // info["n_heads"]
    else:
        info["d_head"] = None

    # Normalization
    if hasattr(config, "rms_norm_eps"):
        info["normalization_type"] = "RMS"
        info["eps"] = config.rms_norm_eps
        info["eps_attr_candidates"] = ["variance_epsilon", "rms_norm_eps"]
    elif hasattr(config, "layer_norm_eps"):
        info["normalization_type"] = "LN"
        info["eps"] = config.layer_norm_eps
        info["eps_attr_candidates"] = ["layer_norm_eps", "eps"]
    elif hasattr(config, "layer_norm_epsilon"):
        info["normalization_type"] = "LN"
        info["eps"] = config.layer_norm_epsilon
        info["eps_attr_candidates"] = ["layer_norm_epsilon", "eps"]
    else:
        info["normalization_type"] = "unknown"
        info["eps_attr_candidates"] = []

    # Attention type
    n_heads = info["n_heads"]
    n_kv_heads = info["n_key_value_heads"]
    if n_kv_heads is None or n_kv_heads == n_heads:
        info["attention_type"] = "MHA"
    elif n_kv_heads == 1:
        info["attention_type"] = "MQA"
    else:
        info["attention_type"] = "GQA"

    # Positional embedding type
    rope_type = getattr(config, "rope_scaling", None)
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None or rope_type is not None:
        info["positional_embedding_type"] = "rotary"
    elif hasattr(config, "position_embedding_type"):
        info["positional_embedding_type"] = config.position_embedding_type
    else:
        # Check for max_position_embeddings without rope indicators
        info["positional_embedding_type"] = "standard (inferred — verify manually)"

    # MLP type (heuristic: if intermediate_size != 4 * hidden_size, likely gated)
    if hasattr(config, "mlp_bias"):
        info["mlp_has_bias"] = config.mlp_bias
    if hasattr(config, "hidden_act"):
        info["activation"] = config.hidden_act
        if config.hidden_act in ("silu", "swiglu"):
            info["mlp_type"] = "gated (SwiGLU)"
        elif config.hidden_act in ("gelu", "gelu_new", "relu"):
            info["mlp_type"] = "standard (inferred from activation — verify manually)"
        else:
            info["mlp_type"] = f"unknown (activation: {config.hidden_act})"

    # Biases
    if hasattr(config, "attention_bias"):
        info["attention_has_bias"] = config.attention_bias
    if hasattr(config, "bias"):
        info["global_bias"] = config.bias

    return info


def suggest_closest_adapter(info: dict) -> str:
    """Suggest the closest existing adapter to use as a reference."""
    norm = info.get("normalization_type", "")
    pos = info.get("positional_embedding_type", "")
    mlp = info.get("mlp_type", "")
    attn = info.get("attention_type", "")

    if "rotary" in pos and "RMS" in norm and "gated" in mlp:
        if attn == "GQA":
            return "llama.py (Llama-like with GQA)"
        return "llama.py (Llama-like)"
    elif "rotary" in pos and "RMS" in norm:
        return "qwen2.py (RoPE + RMSNorm — uses MLPBridge instead of GatedMLPBridge)"
    elif "standard" in pos and "LN" in norm:
        return "gpt2.py (GPT-2-like)"
    else:
        return "llama.py (default starting point — verify manually)"


def analyze_modules(model) -> list[str]:
    """List all named modules in the model."""
    modules = []
    for name, module in model.named_modules():
        modules.append(f"{name}: {type(module).__name__}")
    return modules


def detect_module_paths(model) -> dict:
    """Detect HF module paths for component mapping by inspecting the model tree."""
    paths = {}
    module_map = {name: type(mod).__name__ for name, mod in model.named_modules()}

    # Find embedding
    for name, cls in module_map.items():
        if cls == "Embedding" and "embed" in name.lower() and "pos" not in name.lower():
            paths["embed"] = name
            break

    # Find positional embedding
    for name, cls in module_map.items():
        if cls == "Embedding" and ("pos" in name.lower() or "wpe" in name.lower()):
            paths["pos_embed"] = name
            break

    # Find rotary embedding
    for name, cls in module_map.items():
        if "RotaryEmbedding" in cls and "layers" not in name:
            paths["rotary_emb"] = name
            break

    # Find layer list (blocks)
    for name, cls in module_map.items():
        if cls == "ModuleList" and any(k in name for k in ["layers", ".h", "blocks"]):
            paths["blocks"] = name
            break

    # Find final layer norm
    for name, cls in module_map.items():
        if ("Norm" in cls or "LayerNorm" in cls) and name.count(".") <= 1:
            if "layers" not in name and "block" not in name and "h." not in name:
                if "final" in name or name.endswith("norm") or name.endswith("ln_f"):
                    paths["ln_final"] = name
                    break

    # Find lm_head / unembed
    for name, cls in module_map.items():
        if cls == "Linear" and ("lm_head" in name or "embed_out" in name):
            paths["unembed"] = name
            break

    # Find block submodule paths (from layer 0)
    blocks_prefix = paths.get("blocks", "")
    if blocks_prefix:
        layer0_prefix = f"{blocks_prefix}.0."
        layer0_modules = {
            name[len(layer0_prefix) :]: cls
            for name, cls in module_map.items()
            if name.startswith(layer0_prefix) and name.count(".") - layer0_prefix.count(".") <= 1
        }

        # Attention module
        for name, cls in layer0_modules.items():
            if "attn" in name.lower() or "attention" in name.lower():
                paths["attn"] = name
                # Find Q/K/V/O projections within attention
                attn_prefix = f"{layer0_prefix}{name}."
                attn_children = {
                    n[len(attn_prefix) :]: c
                    for n, c in module_map.items()
                    if n.startswith(attn_prefix) and n.count(".") == attn_prefix.count(".")
                }
                for proj_name, proj_cls in attn_children.items():
                    if proj_cls == "Linear":
                        if "q" in proj_name.lower() and "k" not in proj_name.lower():
                            paths["attn_q"] = proj_name
                        elif "k" in proj_name.lower() and "q" not in proj_name.lower():
                            paths["attn_k"] = proj_name
                        elif "v" in proj_name.lower():
                            paths["attn_v"] = proj_name
                        elif (
                            "o" in proj_name.lower()
                            or "out" in proj_name.lower()
                            or proj_name == "dense"
                        ):
                            paths["attn_o"] = proj_name
                        elif "c_attn" in proj_name or "qkv" in proj_name.lower():
                            paths["attn_qkv"] = proj_name
                        elif "c_proj" in proj_name:
                            paths["attn_o"] = proj_name
                break

        # MLP module
        for name, cls in layer0_modules.items():
            if "mlp" in name.lower() or "feed_forward" in name.lower():
                paths["mlp"] = name
                mlp_prefix = f"{layer0_prefix}{name}."
                mlp_children = {
                    n[len(mlp_prefix) :]: c
                    for n, c in module_map.items()
                    if n.startswith(mlp_prefix) and n.count(".") == mlp_prefix.count(".")
                }
                for proj_name, proj_cls in mlp_children.items():
                    if proj_cls in ("Linear", "Conv1D"):
                        pn = proj_name.lower()
                        if "gate" in pn or pn == "w1":
                            paths["mlp_gate"] = proj_name
                        elif "up" in pn or pn == "w3" or "c_fc" in pn or "fc1" in pn:
                            paths["mlp_in"] = proj_name
                        elif "down" in pn or pn == "w2" or "fc2" in pn:
                            paths["mlp_out"] = proj_name
                        elif "c_proj" in pn:
                            paths["mlp_out"] = proj_name
                break

        # Layer norms
        for name, cls in layer0_modules.items():
            if "Norm" in cls or "LayerNorm" in cls:
                if "input" in name or "ln_1" in name or name == "ln1":
                    paths["ln1"] = name
                elif "post" in name or "ln_2" in name or name == "ln2":
                    paths["ln2"] = name

    return paths


def generate_scaffold(info: dict, paths: dict, state_dict_keys: list) -> str:
    """Generate a pre-filled adapter Python file."""
    arch_class = info["architecture_class"]
    # Derive class name: strip For*LM suffix, add ArchitectureAdapter
    base_name = re.sub(r"(For(Causal|Conditional|Masked).*|LMHead.*)", "", arch_class)
    adapter_class = f"{base_name}ArchitectureAdapter"
    module_name = base_name.lower()

    norm_type = info.get("normalization_type", "unknown")
    pos_type = info.get("positional_embedding_type", "unknown")
    is_rope = "rotary" in pos_type
    is_rms = norm_type == "RMS"
    is_gated = "gated" in info.get("mlp_type", "").lower()
    has_combined_qkv = "attn_qkv" in paths
    has_gqa = info.get("attention_type") in ("GQA", "MQA")

    # Build imports
    norm_bridge = "RMSNormalizationBridge" if is_rms else "NormalizationBridge"
    attn_bridge = (
        "JointQKVAttentionBridge" if has_combined_qkv else "PositionEmbeddingsAttentionBridge"
    )
    mlp_bridge = "GatedMLPBridge" if is_gated else "MLPBridge"
    pos_bridge = "RotaryEmbeddingBridge" if is_rope else "PosEmbedBridge"

    imports = [
        "BlockBridge",
        "EmbeddingBridge",
        mlp_bridge,
        "LinearBridge",
        attn_bridge,
        norm_bridge,
        pos_bridge,
        "UnembeddingBridge",
    ]

    # Determine eps_attr
    eps_attr = "variance_epsilon"
    candidates = info.get("eps_attr_candidates", [])
    if candidates:
        eps_attr = candidates[0]

    # Detect bias presence from state_dict keys
    has_attn_bias = any(
        k.endswith(".q_proj.bias") or k.endswith(".c_attn.bias") for k in state_dict_keys
    )
    has_mlp_bias = any(
        k.endswith(".up_proj.bias") or k.endswith(".c_fc.bias") or k.endswith(".fc1.bias")
        for k in state_dict_keys
    )

    # Build optional params docstring
    optional_params = []
    if not has_attn_bias:
        optional_params.extend(
            [
                "- blocks.{i}.attn.b_Q - No bias on query projection",
                "- blocks.{i}.attn.b_K - No bias on key projection",
                "- blocks.{i}.attn.b_V - No bias on value projection",
                "- blocks.{i}.attn.b_O - No bias on output projection",
            ]
        )
    if not has_mlp_bias:
        gate_line = (
            "- blocks.{i}.mlp.b_gate - No bias on MLP gate projection\n    " if is_gated else ""
        )
        optional_params.extend(
            [
                "- blocks.{i}.mlp.b_in - No bias on MLP input",
                f"{gate_line}- blocks.{{i}}.mlp.b_out - No bias on MLP output",
            ]
        )
    if is_rms:
        optional_params.extend(
            [
                "- blocks.{i}.ln1.b - RMSNorm has no bias",
                "- blocks.{i}.ln2.b - RMSNorm has no bias",
                "- ln_final.b - RMSNorm has no bias",
            ]
        )

    optional_block = (
        "\n    ".join(optional_params) if optional_params else "None identified — verify manually"
    )

    # Build component mapping
    embed_path = paths.get("embed", "model.embed_tokens  # TODO: verify")
    blocks_path = paths.get("blocks", "model.layers  # TODO: verify")
    ln1_path = paths.get("ln1", "input_layernorm  # TODO: verify")
    ln2_path = paths.get("ln2", "post_attention_layernorm  # TODO: verify")
    attn_path = paths.get("attn", "self_attn  # TODO: verify")
    mlp_path = paths.get("mlp", "mlp  # TODO: verify")
    ln_final_path = paths.get("ln_final", "model.norm  # TODO: verify")
    unembed_path = paths.get("unembed", "lm_head  # TODO: verify")

    # Attention submodules
    if has_combined_qkv:
        attn_submodules = f"""                            "qkv": LinearBridge(name="{paths.get('attn_qkv', 'c_attn')}"),
                            "o": LinearBridge(name="{paths.get('attn_o', 'c_proj')}"),"""
    else:
        attn_submodules = f"""                            "q": LinearBridge(name="{paths.get('attn_q', 'q_proj')}"),
                            "k": LinearBridge(name="{paths.get('attn_k', 'k_proj')}"),
                            "v": LinearBridge(name="{paths.get('attn_v', 'v_proj')}"),
                            "o": LinearBridge(name="{paths.get('attn_o', 'o_proj')}"),"""

    # MLP submodules
    if is_gated:
        mlp_submodules = f"""                            "gate": LinearBridge(name="{paths.get('mlp_gate', 'gate_proj')}"),
                            "in": LinearBridge(name="{paths.get('mlp_in', 'up_proj')}"),
                            "out": LinearBridge(name="{paths.get('mlp_out', 'down_proj')}"),"""
    else:
        mlp_submodules = f"""                            "in": LinearBridge(name="{paths.get('mlp_in', 'c_fc')}"),
                            "out": LinearBridge(name="{paths.get('mlp_out', 'c_proj')}"),"""

    # Positional embedding component
    if is_rope:
        pos_component = f"""        "rotary_emb": RotaryEmbeddingBridge(name="{paths.get('rotary_emb', 'model.rotary_emb  # TODO: verify')}"),"""
        attn_kwargs = """
                        requires_attention_mask=True,
                        requires_position_embeddings=True,"""
    else:
        pos_component = f"""        "pos_embed": PosEmbedBridge(name="{paths.get('pos_embed', 'transformer.wpe  # TODO: verify')}"),"""
        attn_kwargs = ""

    # GQA handling
    gqa_block = ""
    if has_gqa:
        gqa_block = """
        # GQA support
        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads
"""

    # Setup component testing (for RoPE models)
    setup_testing = ""
    if is_rope:
        rotary_path = paths.get("rotary_emb", "model.rotary_emb")
        # Derive the Python access path from the dot-separated path
        rotary_access = "hf_model." + rotary_path
        setup_testing = f"""
    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        \"\"\"Set up rotary embedding references for component testing.\"\"\"
        rotary_emb = {rotary_access}

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
"""

    # Weight conversions
    weight_conv = "**self._qkvo_weight_conversions(),"
    if has_combined_qkv:
        weight_conv = """# TODO: Combined QKV requires custom weight conversions.
            # See gpt2.py QKVSplitRearrangeConversion for reference.
            # Standard _qkvo_weight_conversions() does NOT work for combined QKV."""

    scaffold = f'''"""{base_name} architecture adapter.

Generated by analyze-hf-model.py --scaffold from {info.get("_model_id", "unknown model")}.
Review all TODO comments and verify paths against the Transformers source.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    {(",{chr(10)}    ").join(sorted(set(imports)))}
)


class {adapter_class}(ArchitectureAdapter):
    """{base_name} architecture adapter.

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    {optional_block}
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the {base_name} architecture adapter."""
        super().__init__(cfg)

        # Config attributes (detected from model config)
        self.cfg.normalization_type = "{norm_type}"
        self.cfg.positional_embedding_type = "{"rotary" if is_rope else "standard"}"
        self.cfg.final_rms = {is_rms}
        self.cfg.gated_mlp = {is_gated}
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = {is_rms}
        self.cfg.eps_attr = "{eps_attr}"
{gqa_block}
        # Weight processing conversions
        self.weight_processing_conversions = {{
            {weight_conv}
        }}

        # Component mapping (paths detected from model module tree)
        self.component_mapping = {{
            "embed": EmbeddingBridge(name="{embed_path}"),
    {pos_component}
            "blocks": BlockBridge(
                name="{blocks_path}",
                submodules={{
                    "ln1": {norm_bridge}(name="{ln1_path}", config=self.cfg),
                    "ln2": {norm_bridge}(name="{ln2_path}", config=self.cfg),
                    "attn": {attn_bridge}(
                        name="{attn_path}",
                        config=self.cfg,
                        submodules={{
{attn_submodules}
                        }},{attn_kwargs}
                    ),
                    "mlp": {mlp_bridge}(
                        name="{mlp_path}",
                        config=self.cfg,
                        submodules={{
{mlp_submodules}
                        }},
                    ),
                }},
            ),
            "ln_final": {norm_bridge}(name="{ln_final_path}", config=self.cfg),
            "unembed": UnembeddingBridge(name="{unembed_path}", config=self.cfg),
        }}
{setup_testing}'''

    return scaffold


def main():
    parser = argparse.ArgumentParser(description="Analyze HF model for adapter creation")
    parser.add_argument("model", help="HuggingFace model name or path")
    parser.add_argument(
        "--modules",
        action="store_true",
        help="Also print model module hierarchy (requires downloading model)",
    )
    parser.add_argument(
        "--state-dict",
        action="store_true",
        help="Also print state dict keys and shapes (requires downloading model)",
    )
    parser.add_argument(
        "--scaffold",
        action="store_true",
        help="Generate a pre-filled adapter .py file (meta device, no weights)",
    )
    parser.add_argument(
        "--scaffold-out", default=None, help="Write scaffold to this file path (default: stdout)"
    )
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()

    try:
        from transformers import AutoConfig
    except ImportError:
        print(
            "Error: transformers package not installed. Run: pip install transformers",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load config (doesn't download the model)
    print(f"Loading config for: {args.model}", file=sys.stderr)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    info = analyze_config(config)
    info["suggested_adapter"] = suggest_closest_adapter(info)

    if args.scaffold:
        import torch
        from transformers import AutoModelForCausalLM

        print("Creating model on meta device (no weights downloaded)...", file=sys.stderr)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        state_dict_keys = list(model.state_dict().keys())
        paths = detect_module_paths(model)
        info["_model_id"] = args.model

        print(
            f"Detected {len(paths)} module paths, {len(state_dict_keys)} state_dict keys",
            file=sys.stderr,
        )

        scaffold_code = generate_scaffold(info, paths, state_dict_keys)

        if args.scaffold_out:
            Path(args.scaffold_out).write_text(scaffold_code)
            print(f"Scaffold written to: {args.scaffold_out}", file=sys.stderr)
        else:
            print(scaffold_code)
        sys.exit(0)

    if args.modules or args.state_dict:
        import torch
        from transformers import AutoModelForCausalLM

        print("Loading model (this may take a while)...", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu",
        )

        if args.modules:
            info["modules"] = analyze_modules(model)

        if args.state_dict:
            info["state_dict_keys"] = [
                {"key": k, "shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in model.state_dict().items()
            ]

    if args.json:
        print(json.dumps(info, indent=2, default=str))
    else:
        print("\n" + "=" * 60)
        print(f"  Architecture Adapter Analysis: {args.model}")
        print("=" * 60)
        print(f"\n  Architecture class : {info['architecture_class']}")
        print(f"  Model type         : {info['model_type']}")
        print(f"\n  --- Dimensions ---")
        print(f"  d_model            : {info.get('d_model')}")
        print(f"  n_heads            : {info.get('n_heads')}")
        print(f"  n_key_value_heads  : {info.get('n_key_value_heads')}")
        print(f"  d_head             : {info.get('d_head')}")
        print(f"  d_mlp              : {info.get('d_mlp')}")
        print(f"  n_layers           : {info.get('n_layers')}")
        print(f"  d_vocab            : {info.get('d_vocab')}")
        print(f"  n_ctx              : {info.get('n_ctx')}")
        print(f"\n  --- Architecture ---")
        print(f"  Normalization      : {info.get('normalization_type')}")
        print(f"  Epsilon attr       : {info.get('eps_attr_candidates')}")
        print(f"  Pos embeddings     : {info.get('positional_embedding_type')}")
        print(f"  Attention type     : {info.get('attention_type')}")
        print(f"  MLP type           : {info.get('mlp_type', 'unknown')}")
        print(f"  Activation         : {info.get('activation', 'unknown')}")
        print(f"  Attention bias     : {info.get('attention_has_bias', 'unknown')}")
        print(f"\n  --- Recommendation ---")
        print(f"  Closest adapter    : {info['suggested_adapter']}")

        if "modules" in info:
            print(f"\n  --- Module Hierarchy ---")
            for m in info["modules"][:100]:
                print(f"    {m}")
            if len(info["modules"]) > 100:
                print(f"    ... ({len(info['modules']) - 100} more)")

        if "state_dict_keys" in info:
            print(f"\n  --- State Dict Keys ({len(info['state_dict_keys'])} params) ---")
            for entry in info["state_dict_keys"][:50]:
                print(f"    {entry['key']}: {entry['shape']}")
            if len(info["state_dict_keys"]) > 50:
                print(f"    ... ({len(info['state_dict_keys']) - 50} more)")

        print()


if __name__ == "__main__":
    main()
