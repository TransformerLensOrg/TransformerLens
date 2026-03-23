#!/usr/bin/env python3
"""Generate a markdown report of supported and unsupported models.

This script generates a comprehensive report showing:
- All supported model IDs grouped by architecture
- Total count of supported models
- Unsupported architectures with model counts and descriptions

Usage:
    python -m transformer_lens.tools.model_registry.generate_report
    python -m transformer_lens.tools.model_registry.generate_report --output custom_report.md
    python -m transformer_lens.tools.model_registry.generate_report --help
"""

import argparse
from datetime import datetime
from pathlib import Path

from .api import (
    get_registry_stats,
    get_supported_architectures,
    get_supported_models,
    get_unsupported_architectures,
)

# Descriptions of common architectures (both supported and unsupported)
ARCHITECTURE_DESCRIPTIONS: dict[str, str] = {
    # Supported architectures
    "GPT2LMHeadModel": "OpenAI's GPT-2 decoder-only transformer for causal language modeling",
    "GPTNeoForCausalLM": "EleutherAI's GPT-Neo, an open-source GPT-3-like model",
    "GPTNeoXForCausalLM": "EleutherAI's GPT-NeoX architecture used in Pythia models",
    "GPTJForCausalLM": "EleutherAI's GPT-J 6B parameter model",
    "LlamaForCausalLM": "Meta's LLaMA architecture, basis for many open models",
    "MistralForCausalLM": "Mistral AI's efficient 7B parameter model with sliding window attention",
    "MixtralForCausalLM": "Mistral AI's Mixture of Experts model",
    "GemmaForCausalLM": "Google's Gemma lightweight open model family",
    "Gemma2ForCausalLM": "Google's Gemma 2 with improved architecture",
    "Gemma3ForCausalLM": "Google's Gemma 3 latest generation",
    "Qwen2ForCausalLM": "Alibaba's Qwen2 multilingual model",
    "Qwen3ForCausalLM": "Alibaba's Qwen3 latest generation",
    "BloomForCausalLM": "BigScience's BLOOM multilingual model",
    "OPTForCausalLM": "Meta's Open Pre-trained Transformer",
    "PhiForCausalLM": "Microsoft's Phi small language model",
    "Phi3ForCausalLM": "Microsoft's Phi-3 improved small model",
    "FalconForCausalLM": "TII's Falcon model series",
    "OlmoForCausalLM": "Allen AI's OLMo open language model",
    "Olmo2ForCausalLM": "Allen AI's OLMo 2 with improved training",
    "Olmo3ForCausalLM": "Allen AI's OLMo 3 latest generation",
    "OlmoeForCausalLM": "Allen AI's OLMoE Mixture of Experts model",
    "StableLmForCausalLM": "Stability AI's StableLM model",
    "T5ForConditionalGeneration": "Google's T5 encoder-decoder model (partial support)",
    # Unsupported architectures
    "BertModel": "Google's BERT bidirectional encoder for understanding tasks",
    "BertForMaskedLM": "BERT with masked language modeling head",
    "BertForSequenceClassification": "BERT fine-tuned for classification",
    "RobertaModel": "Facebook's RoBERTa, optimized BERT training",
    "RobertaForMaskedLM": "RoBERTa with masked language modeling head",
    "DistilBertModel": "Distilled version of BERT, 40% smaller",
    "AlbertModel": "A Lite BERT with parameter sharing",
    "XLNetLMHeadModel": "Google/CMU's XLNet with permutation language modeling",
    "ElectraModel": "Google's ELECTRA with replaced token detection",
    "DebertaModel": "Microsoft's DeBERTa with disentangled attention",
    "DebertaV2Model": "DeBERTa version 2 with improved architecture",
    "MPNetModel": "Microsoft's MPNet combining MLM and PLM",
    "LongformerModel": "Allen AI's Longformer for long documents",
    "BigBirdModel": "Google's BigBird with sparse attention",
    "ReformerModel": "Google's Reformer with locality-sensitive hashing",
    "BartForConditionalGeneration": "Facebook's BART encoder-decoder model",
    "MBartForConditionalGeneration": "Multilingual BART",
    "PegasusForConditionalGeneration": "Google's PEGASUS for summarization",
    "MT5ForConditionalGeneration": "Multilingual T5",
    "WhisperForConditionalGeneration": "OpenAI's Whisper speech recognition",
    "CLIPModel": "OpenAI's CLIP vision-language model",
    "ViTModel": "Google's Vision Transformer",
    "SwinModel": "Microsoft's Swin Transformer for vision",
    "DeiTModel": "Facebook's Data-efficient Image Transformer",
    "BeitModel": "Microsoft's BERT pre-training for images",
    "ConvNextModel": "Facebook's ConvNeXt modernized ConvNet",
    "SegformerModel": "NVIDIA's SegFormer for segmentation",
    "Wav2Vec2Model": "Facebook's Wav2Vec 2.0 for speech",
    "HubertModel": "Facebook's HuBERT for speech",
    "SpeechT5Model": "Microsoft's SpeechT5 for speech tasks",
    "BlipModel": "Salesforce's BLIP vision-language model",
    "Blip2Model": "Salesforce's BLIP-2 with frozen LLM",
    "LlavaForConditionalGeneration": "Visual instruction-tuned LLaMA",
    "GitModel": "Microsoft's GIT for vision-language",
    "PaliGemmaForConditionalGeneration": "Google's PaliGemma vision-language",
    "CohereForCausalLM": "Cohere's Command models",
    "DeepseekForCausalLM": "DeepSeek's open models",
    "InternLMForCausalLM": "Shanghai AI Lab's InternLM",
    "BaichuanForCausalLM": "Baichuan's Chinese-focused models",
    "YiForCausalLM": "01.AI's Yi model series",
    "OrionForCausalLM": "OrionStar's Orion models",
    "StarcoderForCausalLM": "BigCode's StarCoder for code",
    "CodeLlamaForCausalLM": "Meta's Code Llama for programming",
    "CodeGenForCausalLM": "Salesforce's CodeGen models",
    "SantacoderForCausalLM": "BigCode's SantaCoder",
}


def get_architecture_description(arch_id: str) -> str:
    """Get a description for an architecture, with fallback."""
    if arch_id in ARCHITECTURE_DESCRIPTIONS:
        return ARCHITECTURE_DESCRIPTIONS[arch_id]

    # Generate a basic description from the name
    if "ForCausalLM" in arch_id:
        base = arch_id.replace("ForCausalLM", "")
        return f"{base} architecture for causal language modeling"
    elif "ForConditionalGeneration" in arch_id:
        base = arch_id.replace("ForConditionalGeneration", "")
        return f"{base} encoder-decoder for conditional generation"
    elif "ForMaskedLM" in arch_id:
        base = arch_id.replace("ForMaskedLM", "")
        return f"{base} with masked language modeling head"
    elif "ForSequenceClassification" in arch_id:
        base = arch_id.replace("ForSequenceClassification", "")
        return f"{base} fine-tuned for sequence classification"
    elif "Model" in arch_id:
        base = arch_id.replace("Model", "")
        return f"{base} base model architecture"
    else:
        return "Transformer architecture"


def generate_report(output_path: Path | None = None) -> str:
    """Generate the markdown report.

    Args:
        output_path: Optional path to write the report. If None, only returns the string.

    Returns:
        The generated markdown report as a string.
    """
    # Gather data
    models = get_supported_models()
    architectures = get_supported_architectures()
    gaps = get_unsupported_architectures()
    stats = get_registry_stats()

    # Group models by architecture
    models_by_arch: dict[str, list[str]] = {}
    for model in models:
        arch = model.architecture_id
        if arch not in models_by_arch:
            models_by_arch[arch] = []
        models_by_arch[arch].append(model.model_id)

    # Sort models within each architecture
    for arch in models_by_arch:
        models_by_arch[arch].sort()

    # Calculate totals
    total_supported = len(models)
    total_unsupported = sum(g.total_models for g in gaps)
    total_all = total_supported + total_unsupported

    # Build report
    lines = []
    lines.append("# TransformerLens Model Compatibility Report")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Supported Models | {total_supported:,} |")
    lines.append(f"| Supported Architectures | {len(architectures)} |")
    lines.append(f"| Verified Models | {stats['total_verified']} |")
    lines.append(f"| Unsupported Architectures | {len(gaps)} |")
    lines.append(f"| Models in Unsupported Architectures | {total_unsupported:,} |")
    lines.append(f"| **Total Potential Models** | **{total_all:,}** |")
    lines.append("")

    # Supported models section
    lines.append("## Supported Models")
    lines.append("")
    lines.append(
        f"TransformerLens supports **{total_supported:,} models** across **{len(architectures)} architectures**."
    )
    lines.append("")

    for arch in sorted(models_by_arch.keys()):
        model_list = models_by_arch[arch]
        desc = get_architecture_description(arch)
        lines.append(f"### {arch}")
        lines.append("")
        lines.append(f"*{desc}*")
        lines.append("")
        lines.append(f"**{len(model_list)} models:**")
        lines.append("")
        for model_id in model_list:
            # Check if verified
            model_entry = next((m for m in models if m.model_id == model_id), None)
            verified_badge = " ✓" if model_entry and model_entry.status == 1 else ""
            lines.append(f"- `{model_id}`{verified_badge}")
        lines.append("")

    # Unsupported architectures section
    lines.append("## Unsupported Architectures")
    lines.append("")
    lines.append(
        f"The following **{len(gaps)} architectures** are not yet supported by TransformerLens,"
    )
    lines.append(f"representing **{total_unsupported:,} models** on HuggingFace.")
    lines.append("")
    lines.append("| Architecture | Models | Description |")
    lines.append("|--------------|--------|-------------|")

    for gap in gaps:
        desc = get_architecture_description(gap.architecture_id)
        lines.append(f"| `{gap.architecture_id}` | {gap.total_models:,} | {desc} |")

    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append(
        "*Report generated by `python -m transformer_lens.tools.model_registry.generate_report`*"
    )
    lines.append("")
    lines.append("✓ = Verified to work with TransformerLens")

    report = "\n".join(lines)

    # Write to file if path provided
    if output_path:
        output_path.write_text(report)
        print(f"Report written to: {output_path}")

    return report


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate a markdown report of TransformerLens model compatibility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate report to default location (MODEL_COMPATIBILITY_REPORT.md)
    python -m transformer_lens.tools.model_registry.generate_report

    # Generate report to custom location
    python -m transformer_lens.tools.model_registry.generate_report -o my_report.md

    # Print report to stdout only
    python -m transformer_lens.tools.model_registry.generate_report --stdout
""",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: MODEL_COMPATIBILITY_REPORT.md in current directory)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print report to stdout instead of writing to file",
    )

    args = parser.parse_args()

    if args.stdout:
        report = generate_report()
        print(report)
    else:
        output_path = args.output or Path("MODEL_COMPATIBILITY_REPORT.md")
        generate_report(output_path)


if __name__ == "__main__":
    main()
