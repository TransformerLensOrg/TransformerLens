"""Architecture adapter factory.

This module provides a factory for creating architecture adapters, including
support for external registration and entry-point discovery.
"""

import warnings
from importlib.metadata import entry_points

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.supported_architectures import (
    AfmoeArchitectureAdapter,
    ApertusArchitectureAdapter,
    ArceeArchitectureAdapter,
    AudioFlamingo3ArchitectureAdapter,
    BaichuanArchitectureAdapter,
    BambaArchitectureAdapter,
    BartArchitectureAdapter,
    BD3LMArchitectureAdapter,
    BertArchitectureAdapter,
    BitNetArchitectureAdapter,
    BlenderbotArchitectureAdapter,
    BloomArchitectureAdapter,
    CodeGenArchitectureAdapter,
    Cohere2ArchitectureAdapter,
    CohereArchitectureAdapter,
    DeepSeekV2ArchitectureAdapter,
    DeepSeekV3ArchitectureAdapter,
    DeepSeekV4ArchitectureAdapter,
    DreamArchitectureAdapter,
    Emu3ArchitectureAdapter,
    Ernie4_5_MoeArchitectureAdapter,
    Ernie4_5ArchitectureAdapter,
    Exaone4ArchitectureAdapter,
    ExaoneArchitectureAdapter,
    FalconArchitectureAdapter,
    FalconH1ArchitectureAdapter,
    FalconMambaArchitectureAdapter,
    FlexOlmoArchitectureAdapter,
    Florence2ArchitectureAdapter,
    Gemma1ArchitectureAdapter,
    Gemma2ArchitectureAdapter,
    Gemma3ArchitectureAdapter,
    Gemma3MultimodalArchitectureAdapter,
    Gemma3nArchitectureAdapter,
    Gemma4ArchitectureAdapter,
    Gemma4TextArchitectureAdapter,
    GiddArchitectureAdapter,
    Glm4ArchitectureAdapter,
    Glm4MoeArchitectureAdapter,
    Glm4MoeLiteArchitectureAdapter,
    Glm4vArchitectureAdapter,
    GlmArchitectureAdapter,
    GlmAsrArchitectureAdapter,
    GlmMoeDsaArchitectureAdapter,
    GPT2ArchitectureAdapter,
    Gpt2LmHeadCustomArchitectureAdapter,
    GPTBigCodeArchitectureAdapter,
    GptjArchitectureAdapter,
    GPTOSSArchitectureAdapter,
    GraniteArchitectureAdapter,
    GraniteMoeArchitectureAdapter,
    GraniteMoeHybridArchitectureAdapter,
    HrmTextArchitectureAdapter,
    HubertArchitectureAdapter,
    HunYuanDenseV1ArchitectureAdapter,
    HyenaDNAArchitectureAdapter,
    Idefics3ArchitectureAdapter,
    InternLM2ArchitectureAdapter,
    Jais2ArchitectureAdapter,
    JetMoeArchitectureAdapter,
    LagunaArchitectureAdapter,
    LEDArchitectureAdapter,
    Lfm2ArchitectureAdapter,
    Lfm2MoeArchitectureAdapter,
    LLaDA2MoeArchitectureAdapter,
    LLaDAArchitectureAdapter,
    Llama4ArchitectureAdapter,
    Llama4MultimodalArchitectureAdapter,
    LlamaArchitectureAdapter,
    LlavaArchitectureAdapter,
    LlavaNextArchitectureAdapter,
    LlavaOnevisionArchitectureAdapter,
    LongT5ArchitectureAdapter,
    M2M100ArchitectureAdapter,
    Mamba2ArchitectureAdapter,
    MambaArchitectureAdapter,
    MarianArchitectureAdapter,
    MBartArchitectureAdapter,
    MingptArchitectureAdapter,
    MiniMaxM2ArchitectureAdapter,
    Ministral3ArchitectureAdapter,
    Mistral3ArchitectureAdapter,
    MistralArchitectureAdapter,
    MixtralArchitectureAdapter,
    ModernBertDecoderArchitectureAdapter,
    MPTArchitectureAdapter,
    MusicFlamingoArchitectureAdapter,
    NanoChatArchitectureAdapter,
    NanogptArchitectureAdapter,
    NativeArchitectureAdapter,
    NeelSoluOldArchitectureAdapter,
    NemotronArchitectureAdapter,
    NemotronHArchitectureAdapter,
    NeoArchitectureAdapter,
    NeoxArchitectureAdapter,
    Olmo2ArchitectureAdapter,
    Olmo3ArchitectureAdapter,
    OlmoArchitectureAdapter,
    OlmoeArchitectureAdapter,
    OlmoHybridArchitectureAdapter,
    OpenAIGPTArchitectureAdapter,
    OpenElmArchitectureAdapter,
    OptArchitectureAdapter,
    OuroArchitectureAdapter,
    PegasusArchitectureAdapter,
    Phi3ArchitectureAdapter,
    PhiArchitectureAdapter,
    PhiMoEArchitectureAdapter,
    Qwen2_5_VLArchitectureAdapter,
    Qwen2ArchitectureAdapter,
    Qwen2AudioArchitectureAdapter,
    Qwen2MoeArchitectureAdapter,
    Qwen3_5ArchitectureAdapter,
    Qwen3_5MoeArchitectureAdapter,
    Qwen3_5MoeMultimodalArchitectureAdapter,
    Qwen3_5MultimodalArchitectureAdapter,
    Qwen3ArchitectureAdapter,
    Qwen3MoeArchitectureAdapter,
    Qwen3NextArchitectureAdapter,
    Qwen3VLArchitectureAdapter,
    Qwen3VLMoeArchitectureAdapter,
    QwenArchitectureAdapter,
    RecurrentGemmaArchitectureAdapter,
    RWKV7ArchitectureAdapter,
    RwkvArchitectureAdapter,
    SeedOssArchitectureAdapter,
    SmolLM3ArchitectureAdapter,
    StableLmArchitectureAdapter,
    Starcoder2ArchitectureAdapter,
    SwitchTransformersArchitectureAdapter,
    T5ArchitectureAdapter,
    T5Gemma2ArchitectureAdapter,
    T5GemmaArchitectureAdapter,
    VaultGemmaArchitectureAdapter,
    XGLMArchitectureAdapter,
    YoutuArchitectureAdapter,
    Zamba2ArchitectureAdapter,
)

# Export supported architectures
SUPPORTED_ARCHITECTURES = {
    "AfmoeForCausalLM": AfmoeArchitectureAdapter,
    "ApertusForCausalLM": ApertusArchitectureAdapter,
    "ArceeForCausalLM": ArceeArchitectureAdapter,
    "BaiChuanForCausalLM": BaichuanArchitectureAdapter,
    "BaichuanForCausalLM": BaichuanArchitectureAdapter,
    "BambaForCausalLM": BambaArchitectureAdapter,
    "BartForConditionalGeneration": BartArchitectureAdapter,
    "BD3LM": BD3LMArchitectureAdapter,
    "DreamModel": DreamArchitectureAdapter,
    "Emu3ForConditionalGeneration": Emu3ArchitectureAdapter,
    "AudioFlamingo3ForConditionalGeneration": AudioFlamingo3ArchitectureAdapter,
    "FlexOlmoForCausalLM": FlexOlmoArchitectureAdapter,
    "GiddForDiffusionLM": GiddArchitectureAdapter,
    "HyenaDNAForCausalLM": HyenaDNAArchitectureAdapter,
    "LLaDA2MoeModelLM": LLaDA2MoeArchitectureAdapter,
    "Jais2ForCausalLM": Jais2ArchitectureAdapter,
    "JetMoeForCausalLM": JetMoeArchitectureAdapter,
    # jetmoe-8b checkpoints predate the native port and use the remote-code capitalization
    "JetMoEForCausalLM": JetMoeArchitectureAdapter,
    "LagunaForCausalLM": LagunaArchitectureAdapter,
    "Ministral3ForCausalLM": Ministral3ArchitectureAdapter,
    "VaultGemmaForCausalLM": VaultGemmaArchitectureAdapter,
    "YoutuForCausalLM": YoutuArchitectureAdapter,
    "ModernBertDecoderForCausalLM": ModernBertDecoderArchitectureAdapter,
    "MusicFlamingoForConditionalGeneration": MusicFlamingoArchitectureAdapter,
    "NanoChatForCausalLM": NanoChatArchitectureAdapter,
    "RwkvForCausalLM": RwkvArchitectureAdapter,
    "SwitchTransformersForConditionalGeneration": SwitchTransformersArchitectureAdapter,
    "BertForMaskedLM": BertArchitectureAdapter,
    "BertLMHeadModel": BertArchitectureAdapter,
    "BitNetForCausalLM": BitNetArchitectureAdapter,
    "BlenderbotForConditionalGeneration": BlenderbotArchitectureAdapter,
    "BloomForCausalLM": BloomArchitectureAdapter,
    "BloomModel": BloomArchitectureAdapter,
    "CodeGenForCausalLM": CodeGenArchitectureAdapter,
    "Cohere2ForCausalLM": Cohere2ArchitectureAdapter,
    "CohereForCausalLM": CohereArchitectureAdapter,
    "DeepseekV2ForCausalLM": DeepSeekV2ArchitectureAdapter,
    "DeepseekV3ForCausalLM": DeepSeekV3ArchitectureAdapter,
    "Ernie4_5ForCausalLM": Ernie4_5ArchitectureAdapter,
    "Ernie4_5_MoeForCausalLM": Ernie4_5_MoeArchitectureAdapter,
    "ExaoneForCausalLM": ExaoneArchitectureAdapter,
    "Exaone4ForCausalLM": Exaone4ArchitectureAdapter,
    "DeepseekV4ForCausalLM": DeepSeekV4ArchitectureAdapter,
    "FalconForCausalLM": FalconArchitectureAdapter,
    "FalconH1ForCausalLM": FalconH1ArchitectureAdapter,
    "FalconMambaForCausalLM": FalconMambaArchitectureAdapter,
    "Florence2ForConditionalGeneration": Florence2ArchitectureAdapter,
    "GemmaForCausalLM": Gemma1ArchitectureAdapter,  # Default to Gemma1 as it's the original version
    "Gemma1ForCausalLM": Gemma1ArchitectureAdapter,
    "Gemma2ForCausalLM": Gemma2ArchitectureAdapter,
    "Gemma3ForCausalLM": Gemma3ArchitectureAdapter,
    "Gemma3ForConditionalGeneration": Gemma3MultimodalArchitectureAdapter,
    "Gemma3nForConditionalGeneration": Gemma3nArchitectureAdapter,
    "Gemma4ForConditionalGeneration": Gemma4ArchitectureAdapter,
    # The unified (encoder-free) 12B variant's text decoder is a strict structural
    # subset of gemma4 (no PLE, no MoE — both optional in the adapter), with the
    # same module paths. Requires transformers >= 5.10 to load.
    "Gemma4UnifiedForConditionalGeneration": Gemma4ArchitectureAdapter,
    "Gemma4ForCausalLM": Gemma4TextArchitectureAdapter,
    "GraniteForCausalLM": GraniteArchitectureAdapter,
    "GraniteMoeForCausalLM": GraniteMoeArchitectureAdapter,
    "GraniteMoeHybridForCausalLM": GraniteMoeHybridArchitectureAdapter,
    "GlmForCausalLM": GlmArchitectureAdapter,
    "Glm4ForCausalLM": Glm4ArchitectureAdapter,
    "Glm4vForConditionalGeneration": Glm4vArchitectureAdapter,
    "GlmMoeDsaForCausalLM": GlmMoeDsaArchitectureAdapter,
    "Glm4MoeForCausalLM": Glm4MoeArchitectureAdapter,
    "Glm4MoeLiteForCausalLM": Glm4MoeLiteArchitectureAdapter,
    "GlmAsrForConditionalGeneration": GlmAsrArchitectureAdapter,
    "GPT2LMHeadModel": GPT2ArchitectureAdapter,
    "GPTBigCodeForCausalLM": GPTBigCodeArchitectureAdapter,
    "GptOssForCausalLM": GPTOSSArchitectureAdapter,
    "GPT2LMHeadCustomModel": Gpt2LmHeadCustomArchitectureAdapter,
    "GPTJForCausalLM": GptjArchitectureAdapter,
    "HrmTextForCausalLM": HrmTextArchitectureAdapter,
    "HubertForCTC": HubertArchitectureAdapter,
    "HubertModel": HubertArchitectureAdapter,
    "HunYuanDenseV1ForCausalLM": HunYuanDenseV1ArchitectureAdapter,
    "Idefics3ForConditionalGeneration": Idefics3ArchitectureAdapter,
    "InternLM2ForCausalLM": InternLM2ArchitectureAdapter,
    "LEDForConditionalGeneration": LEDArchitectureAdapter,
    "LLaDAModelLM": LLaDAArchitectureAdapter,
    "LlamaForCausalLM": LlamaArchitectureAdapter,
    "Llama4ForCausalLM": Llama4ArchitectureAdapter,
    "Llama4ForConditionalGeneration": Llama4MultimodalArchitectureAdapter,
    "LlavaForConditionalGeneration": LlavaArchitectureAdapter,
    "LlavaNextForConditionalGeneration": LlavaNextArchitectureAdapter,
    "LlavaOnevisionForConditionalGeneration": LlavaOnevisionArchitectureAdapter,
    "Lfm2ForCausalLM": Lfm2ArchitectureAdapter,
    "Lfm2MoeForCausalLM": Lfm2MoeArchitectureAdapter,
    "LongT5ForConditionalGeneration": LongT5ArchitectureAdapter,
    "M2M100ForConditionalGeneration": M2M100ArchitectureAdapter,
    "Mamba2ForCausalLM": Mamba2ArchitectureAdapter,
    "MambaForCausalLM": MambaArchitectureAdapter,
    "MarianMTModel": MarianArchitectureAdapter,
    "MBartForConditionalGeneration": MBartArchitectureAdapter,
    "MiniMaxM2ForCausalLM": MiniMaxM2ArchitectureAdapter,
    "NemotronForCausalLM": NemotronArchitectureAdapter,
    "NemotronHForCausalLM": NemotronHArchitectureAdapter,
    "MixtralForCausalLM": MixtralArchitectureAdapter,
    "MistralForCausalLM": MistralArchitectureAdapter,
    "Mistral3ForConditionalGeneration": Mistral3ArchitectureAdapter,
    "MPTForCausalLM": MPTArchitectureAdapter,
    "MptForCausalLM": MPTArchitectureAdapter,
    "NeoForCausalLM": NeoArchitectureAdapter,
    "NeoXForCausalLM": NeoxArchitectureAdapter,
    "NeelSoluOldForCausalLM": NeelSoluOldArchitectureAdapter,
    "OlmoForCausalLM": OlmoArchitectureAdapter,
    "Olmo2ForCausalLM": Olmo2ArchitectureAdapter,
    "Olmo3ForCausalLM": Olmo3ArchitectureAdapter,
    "OlmoeForCausalLM": OlmoeArchitectureAdapter,
    "OlmoHybridForCausalLM": OlmoHybridArchitectureAdapter,
    "OpenAIGPTLMHeadModel": OpenAIGPTArchitectureAdapter,
    "OpenELMForCausalLM": OpenElmArchitectureAdapter,
    "OPTForCausalLM": OptArchitectureAdapter,
    "PegasusForConditionalGeneration": PegasusArchitectureAdapter,
    "OuroForCausalLM": OuroArchitectureAdapter,
    "PhiForCausalLM": PhiArchitectureAdapter,
    "Phi3ForCausalLM": Phi3ArchitectureAdapter,
    "PhiMoEForCausalLM": PhiMoEArchitectureAdapter,
    "QwenForCausalLM": QwenArchitectureAdapter,
    "Qwen2ForCausalLM": Qwen2ArchitectureAdapter,
    "Qwen2_5_VLForConditionalGeneration": Qwen2_5_VLArchitectureAdapter,
    "Qwen2AudioForConditionalGeneration": Qwen2AudioArchitectureAdapter,
    "Qwen2MoeForCausalLM": Qwen2MoeArchitectureAdapter,
    "Qwen3ForCausalLM": Qwen3ArchitectureAdapter,
    "Qwen3VLForConditionalGeneration": Qwen3VLArchitectureAdapter,
    "Qwen3VLMoeForConditionalGeneration": Qwen3VLMoeArchitectureAdapter,
    "Qwen3MoeForCausalLM": Qwen3MoeArchitectureAdapter,
    "Qwen3NextForCausalLM": Qwen3NextArchitectureAdapter,
    "Qwen3_5ForCausalLM": Qwen3_5ArchitectureAdapter,
    "Qwen3_5ForConditionalGeneration": Qwen3_5MultimodalArchitectureAdapter,
    "Qwen3_5MoeForCausalLM": Qwen3_5MoeArchitectureAdapter,
    "Qwen3_5MoeForConditionalGeneration": Qwen3_5MoeMultimodalArchitectureAdapter,
    "RecurrentGemmaForCausalLM": RecurrentGemmaArchitectureAdapter,
    "SeedOssForCausalLM": SeedOssArchitectureAdapter,
    "RWKV7ForCausalLM": RWKV7ArchitectureAdapter,
    "SmolLM3ForCausalLM": SmolLM3ArchitectureAdapter,
    "StableLmForCausalLM": StableLmArchitectureAdapter,
    "Starcoder2ForCausalLM": Starcoder2ArchitectureAdapter,
    "T5ForConditionalGeneration": T5ArchitectureAdapter,
    "MT5ForConditionalGeneration": T5ArchitectureAdapter,
    "T5WithLMHeadModel": T5ArchitectureAdapter,
    "T5GemmaForConditionalGeneration": T5GemmaArchitectureAdapter,
    "T5Gemma2ForConditionalGeneration": T5Gemma2ArchitectureAdapter,
    "XGLMForCausalLM": XGLMArchitectureAdapter,
    "Zamba2ForCausalLM": Zamba2ArchitectureAdapter,
    "NanoGPTForCausalLM": NanogptArchitectureAdapter,
    "TransformerLensNative": NativeArchitectureAdapter,
    "MinGPTForCausalLM": MingptArchitectureAdapter,
    "GPTNeoForCausalLM": NeoArchitectureAdapter,
    "GPTNeoXForCausalLM": NeoxArchitectureAdapter,
}


class ArchitectureAdapterFactory:
    """Factory for creating architecture adapters.

    Supports external registration via `register_adapter()` and automatic
    discovery of adapters from installed packages via entry points.
    """

    _adapters = dict(SUPPORTED_ARCHITECTURES)
    _entry_points_discovered = False

    @classmethod
    def register_adapter(
        cls, architecture_name: str, adapter_class: type["ArchitectureAdapter"]
    ) -> None:
        """Register a custom architecture adapter at runtime.

        This allows users to add their own architecture adapters without
        modifying TransformerLens source code.

        Args:
            architecture_name: The HuggingFace architecture class name
                (e.g. ``"Qwen3ForCausalLM"``).
            adapter_class: The adapter class to register.

        Example:
            >>> from transformer_lens.config import TransformerBridgeConfig
            >>> from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
            >>> from transformer_lens.factories.architecture_adapter_factory import ArchitectureAdapterFactory
            >>> class MyAdapter(ArchitectureAdapter):
            ...     def __init__(self, cfg):
            ...         super().__init__(cfg)
            >>> ArchitectureAdapterFactory.register_adapter("MyModelForCausalLM", MyAdapter)
            >>> cfg = TransformerBridgeConfig(
            ...     d_model=512, d_head=64, n_layers=6, n_ctx=1024,
            ...     architecture="MyModelForCausalLM",
            ... )
            >>> adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
            >>> isinstance(adapter, MyAdapter)
            True
        """
        cls._adapters[architecture_name] = adapter_class

    @classmethod
    def discover_entry_points(cls) -> None:
        """Discover and register architecture adapters from installed packages.

        Packages can declare adapters in their ``pyproject.toml``:
        ```toml
        [project.entry-points."transformer_lens.architectures"]
        "MyModelForCausalLM" = "my_package.adapters:MyArchitectureAdapter"
        ```
        """
        if cls._entry_points_discovered:
            return
        try:
            eps = entry_points(group="transformer_lens.architectures")
        except Exception as e:
            warnings.warn(
                f"Failed to discover entry points: {e}. " f"External adapters may not be available."
            )
        else:
            for ep in eps:
                try:
                    if ep.name in cls._adapters:
                        dist_name = (
                            getattr(ep.dist, "name", "unknown")
                            if ep.dist is not None
                            else "unknown"
                        )
                        warnings.warn(
                            f"Custom architecture adapter {ep.name} provided by {dist_name} "
                            f"attempted to override a native adapter. If you'd like to use this "
                            f"custom adapter, register it explicitly with register_adapter"
                        )
                        continue
                    cls._adapters[ep.name] = ep.load()
                except Exception as e:
                    warnings.warn(
                        f"Failed to load entry point '{ep.name}': {e}. " f"Skipping this adapter."
                    )
        cls._entry_points_discovered = True

    @classmethod
    def select_architecture_adapter(cls, cfg: TransformerBridgeConfig) -> ArchitectureAdapter:
        """Select the appropriate architecture adapter for the given config.

        Args:
            cfg: The TransformerBridgeConfig to select the adapter for.

        Returns:
            The selected architecture adapter.

        Raises:
            ValueError: If no adapter is found for the given config.
        """
        cls.discover_entry_points()
        if cfg.architecture is not None:
            if cfg.architecture in cls._adapters:
                return cls._adapters[cfg.architecture](cfg)
            else:
                raise ValueError(f"Unsupported architecture: {cfg.architecture}")

        raise ValueError(f"TransformerBridgeConfig must have architecture set, got: {cfg}")
