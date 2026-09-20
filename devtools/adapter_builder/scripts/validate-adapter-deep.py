#!/usr/bin/env python3
"""Deep adapter validation — cross-references adapter against a real HF model.

Loads a model's config and state_dict keys (no weights downloaded) and checks:
1. Component mapping paths resolve to real HF modules
2. Weight conversion keys match actual state_dict parameters
3. No missing or extra entries in weight_processing_conversions

Usage:
    uv run python scripts/validate-adapter-deep.py <architecture_class> <hf_model_id> [--repo <path>]

Example:
    uv run python scripts/validate-adapter-deep.py LlamaForCausalLM meta-llama/Llama-3.2-1B
    uv run python scripts/validate-adapter-deep.py CodeGenForCausalLM Salesforce/codegen-350M-mono
"""

import argparse
import json
import sys
from pathlib import Path


def validate_component_paths(adapter, model):
    """Check that component mapping paths resolve to real HF modules."""
    issues: list[str] = []
    ok: list[str] = []

    if adapter.component_mapping is None:
        issues.append("component_mapping is None")
        return ok, issues

    # Get all named modules from the model
    module_names = {name for name, _ in model.named_modules()}

    for tl_name, component in adapter.component_mapping.items():
        hf_path = component.name
        if hf_path is None:
            continue

        # Top-level components: check the full path
        if hf_path in module_names:
            ok.append(f"{tl_name} -> {hf_path}")
        else:
            # For block components, check with index 0
            test_path = hf_path.replace("{i}", "0") if "{i}" in hf_path else hf_path
            if test_path in module_names:
                ok.append(f"{tl_name} -> {hf_path}")
            else:
                issues.append(f"{tl_name} -> '{hf_path}' does not resolve to a module in the model")

        # Check submodules
        if hasattr(component, "submodules") and component.submodules:
            for sub_name, sub_component in component.submodules.items():
                sub_hf_path = sub_component.name
                if sub_hf_path is None:
                    continue

                # Submodule paths are relative to the parent
                if component.is_list_item if hasattr(component, "is_list_item") else False:
                    full_path = f"{hf_path}.0.{sub_hf_path}"
                else:
                    full_path = f"{hf_path}.{sub_hf_path}"

                if full_path in module_names:
                    ok.append(f"  {tl_name}.{sub_name} -> {sub_hf_path}")
                else:
                    issues.append(
                        f"  {tl_name}.{sub_name} -> '{sub_hf_path}' "
                        f"(full: '{full_path}') does not resolve"
                    )

                # Check nested submodules (e.g., attn.q, mlp.gate)
                if hasattr(sub_component, "submodules") and sub_component.submodules:
                    for nested_name, nested_comp in sub_component.submodules.items():
                        nested_hf = nested_comp.name
                        if nested_hf is None:
                            continue
                        nested_full = f"{hf_path}.0.{sub_hf_path}.{nested_hf}"
                        if nested_full in module_names:
                            ok.append(f"    {tl_name}.{sub_name}.{nested_name} -> {nested_hf}")
                        else:
                            issues.append(
                                f"    {tl_name}.{sub_name}.{nested_name} -> '{nested_hf}' "
                                f"(full: '{nested_full}') does not resolve"
                            )

    return ok, issues


def validate_weight_conversions(adapter, state_dict_keys, n_layers):
    """Check that weight conversion keys match actual state_dict parameters."""
    issues: list[str] = []
    ok: list[str] = []

    if adapter.weight_processing_conversions is None:
        issues.append("weight_processing_conversions is None")
        return ok, issues

    for conv_key, conversion in adapter.weight_processing_conversions.items():
        # Expand {i} to check against actual keys
        if "{i}" in conv_key:
            # Check layer 0 as representative
            expanded = conv_key.replace("{i}", "0")
        else:
            expanded = conv_key

        # Convert TL-style key to what we'd expect in the HF state dict
        # The conversion key is in TL format (blocks.{i}.attn.q.weight)
        # We need to check that the corresponding HF key exists
        # This is a heuristic — the adapter's convert_hf_key_to_tl_key does the mapping
        found = False
        for sd_key in state_dict_keys:
            try:
                tl_key = adapter.convert_hf_key_to_tl_key(sd_key)
                if "{i}" in conv_key:
                    # Normalize layer index for comparison
                    import re

                    pattern = conv_key.replace("{i}", r"\d+")
                    if re.match(pattern.replace(".", r"\."), tl_key):
                        found = True
                        break
                elif tl_key == expanded:
                    found = True
                    break
            except Exception:
                continue

        if found:
            ok.append(f"{conv_key}")
        else:
            # Check if the source_key (if any) maps to something
            source = getattr(conversion, "source_key", None)
            if source:
                issues.append(
                    f"{conv_key} (source: {source}) — no matching HF state_dict key found"
                )
            else:
                issues.append(f"{conv_key} — no matching HF state_dict key found")

    return ok, issues


def check_unmapped_weights(adapter, state_dict_keys, n_layers):
    """Find state_dict keys that have no corresponding weight conversion."""
    unmapped: list[str] = []

    if adapter.weight_processing_conversions is None:
        return unmapped

    # Get all conversion patterns
    conv_patterns = set()
    for conv_key in adapter.weight_processing_conversions:
        if "{i}" in conv_key:
            for i in range(n_layers):
                conv_patterns.add(conv_key.replace("{i}", str(i)))
        else:
            conv_patterns.add(conv_key)

    # Check which attention/projection weights lack conversions
    # (embeddings, norms, etc. typically don't need conversions)
    attention_mlp_keys = [
        k
        for k in state_dict_keys
        if any(
            proj in k
            for proj in [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "query",
                "key",
                "value",
                "c_attn",
                "c_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        )
    ]

    for sd_key in attention_mlp_keys:
        try:
            tl_key = adapter.convert_hf_key_to_tl_key(sd_key)
            if tl_key not in conv_patterns:
                # Check with {i} pattern
                import re

                matched = False
                for conv_key in adapter.weight_processing_conversions:
                    if "{i}" in conv_key:
                        pattern = conv_key.replace("{i}", r"\d+").replace(".", r"\.")
                        if re.match(pattern, tl_key):
                            matched = True
                            break
                if not matched:
                    unmapped.append(f"{sd_key} (TL: {tl_key})")
        except Exception:
            pass

    return unmapped


def main():
    parser = argparse.ArgumentParser(description="Deep adapter validation against a real HF model")
    parser.add_argument("architecture", help="HF architecture class name (e.g., LlamaForCausalLM)")
    parser.add_argument("model_id", help="HuggingFace model ID (e.g., meta-llama/Llama-3.2-1B)")
    parser.add_argument("--repo", default=None, help="Path to TransformerLens repo")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    # Add TL repo to path — the builder lives in devtools/adapter_builder
    # inside the repo, so the containing repo is the default.
    repo = args.repo or str(Path(__file__).resolve().parents[3])
    sys.path.insert(0, repo)

    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        print(
            "Error: transformers and torch required. Run: pip install transformers torch",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )
        from transformer_lens.model_bridge.sources._bridge_builder import (
            build_bridge_config_from_hf,
        )
    except ImportError as e:
        print(f"Error importing TransformerLens: {e}", file=sys.stderr)
        print("Make sure the --repo path is correct.", file=sys.stderr)
        sys.exit(1)

    # Check architecture is registered
    if args.architecture not in SUPPORTED_ARCHITECTURES:
        print(f"Error: {args.architecture} not found in SUPPORTED_ARCHITECTURES", file=sys.stderr)
        print(f"Available: {', '.join(sorted(SUPPORTED_ARCHITECTURES.keys()))}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading config for {args.model_id}...", file=sys.stderr)
    hf_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    # Get state_dict keys without downloading weights
    print(f"Getting state_dict keys (no weight download)...", file=sys.stderr)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
    state_dict_keys = list(model.state_dict().keys())
    n_layers = getattr(hf_config, "num_hidden_layers", 1)

    # Create adapter instance
    print(f"Creating {args.architecture} adapter...", file=sys.stderr)
    try:
        bridge_cfg = build_bridge_config_from_hf(
            hf_config, args.architecture, args.model_id, torch.float32
        )
        adapter_class = SUPPORTED_ARCHITECTURES[args.architecture]
        adapter = adapter_class(bridge_cfg)
    except Exception as e:
        print(f"Error creating adapter: {e}", file=sys.stderr)
        sys.exit(1)

    # Run validations
    print(
        f"Validating against {len(state_dict_keys)} state_dict keys, {n_layers} layers...",
        file=sys.stderr,
    )

    comp_ok, comp_issues = validate_component_paths(adapter, model)
    weight_ok, weight_issues = validate_weight_conversions(adapter, state_dict_keys, n_layers)
    unmapped = check_unmapped_weights(adapter, state_dict_keys, n_layers)

    results = {
        "architecture": args.architecture,
        "model_id": args.model_id,
        "state_dict_keys": len(state_dict_keys),
        "n_layers": n_layers,
        "component_mapping": {"ok": comp_ok, "issues": comp_issues},
        "weight_conversions": {"ok": weight_ok, "issues": weight_issues},
        "unmapped_projection_weights": unmapped,
    }

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        total_issues = len(comp_issues) + len(weight_issues) + len(unmapped)

        print(f"\n{'='*60}")
        print(f"  Deep Validation: {args.architecture}")
        print(f"  Model: {args.model_id}")
        print(f"  State dict keys: {len(state_dict_keys)}, Layers: {n_layers}")
        print(f"{'='*60}\n")

        print("  --- Component Mapping ---")
        for item in comp_ok:
            print(f"  \033[32m[ok]\033[0m {item}")
        for item in comp_issues:
            print(f"  \033[31m[!!]\033[0m {item}")

        print(f"\n  --- Weight Conversions ---")
        for item in weight_ok:
            print(f"  \033[32m[ok]\033[0m {item}")
        for item in weight_issues:
            print(f"  \033[31m[!!]\033[0m {item}")

        if unmapped:
            print(f"\n  --- Unmapped Projection Weights ({len(unmapped)}) ---")
            for item in unmapped[:20]:
                print(f"  \033[33m[??]\033[0m {item}")
            if len(unmapped) > 20:
                print(f"  ... and {len(unmapped) - 20} more")

        print(f"\n{'='*60}")
        if total_issues == 0:
            print(f"  \033[32mAll deep checks passed!\033[0m")
        else:
            print(f"  \033[31m{total_issues} issue(s) found\033[0m")
        print(f"{'='*60}\n")

        sys.exit(1 if total_issues > 0 else 0)


if __name__ == "__main__":
    main()
