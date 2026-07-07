"""Architecture adapter factory.

This module provides a factory for creating architecture adapters, including
support for external registration and entry-point discovery.
"""

import warnings
from importlib.metadata import entry_points

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.supported_architectures import (
    ApertusArchitectureAdapter,
    ArceeArchitectureAdapter,
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
    Ernie4_5ArchitectureAdapter,
    Exaone4ArchitectureAdapter,
    ExaoneArchitectureAdapter,
    FalconArchitectureAdapter,
    FalconH1ArchitectureAdapter,
    FalconMambaArchitectureAdapter,
    Gemma1ArchitectureAdapter,
    Gemma2ArchitectureAdapter,
    Gemma3ArchitectureAdapter,
    Gemma3MultimodalArchitectureAdapter,
    Gemma3nArchitectureAdapter,
    Gemma4ArchitectureAdapter,
    Glm4MoeArchitectureAdapter,
    Glm4MoeLiteArchitectureAdapter,
    GlmArchitectureAdapter,
    GlmMoeDsaArchitectureAdapter,
    GPT2ArchitectureAdapter,
    Gpt2LmHeadCustomArchitectureAdapter,
    GPTBigCodeArchitectureAdapter,
    GptjArchitectureAdapter,
    GPTOSSArchitectureAdapter,
    GraniteArchitectureAdapter,
    GraniteMoeArchitectureAdapter,
    GraniteMoeHybridArchitectureAdapter,
    HubertArchitectureAdapter,
    HunYuanDenseV1ArchitectureAdapter,
    Idefics3ArchitectureAdapter,
    InternLM2ArchitectureAdapter,
    Lfm2MoeArchitectureAdapter,
    LlamaArchitectureAdapter,
    LlavaArchitectureAdapter,
    LlavaNextArchitectureAdapter,
    LlavaOnevisionArchitectureAdapter,
    M2M100ArchitectureAdapter,
    Mamba2ArchitectureAdapter,
    MambaArchitectureAdapter,
    MarianArchitectureAdapter,
    MBartArchitectureAdapter,
    MingptArchitectureAdapter,
    MiniMaxM2ArchitectureAdapter,
    MistralArchitectureAdapter,
    MixtralArchitectureAdapter,
    MPTArchitectureAdapter,
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
    OpenAIGPTArchitectureAdapter,
    OpenElmArchitectureAdapter,
    OptArchitectureAdapter,
    PegasusArchitectureAdapter,
    Phi3ArchitectureAdapter,
    PhiArchitectureAdapter,
    PhiMoEArchitectureAdapter,
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
    QwenArchitectureAdapter,
    RecurrentGemmaArchitectureAdapter,
    SeedOssArchitectureAdapter,
    SmolLM3ArchitectureAdapter,
    StableLmArchitectureAdapter,
    Starcoder2ArchitectureAdapter,
    T5ArchitectureAdapter,
    T5GemmaArchitectureAdapter,
    XGLMArchitectureAdapter,
    Zamba2ArchitectureAdapter,
)

# Export supported architectures
SUPPORTED_ARCHITECTURES = {
    "ApertusForCausalLM": ApertusArchitectureAdapter,
    "ArceeForCausalLM": ArceeArchitectureAdapter,
    "BaiChuanForCausalLM": BaichuanArchitectureAdapter,
    "BaichuanForCausalLM": BaichuanArchitectureAdapter,
    "BambaForCausalLM": BambaArchitectureAdapter,
    "BartForConditionalGeneration": BartArchitectureAdapter,
    "BD3LM": BD3LMArchitectureAdapter,
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
    "ExaoneForCausalLM": ExaoneArchitectureAdapter,
    "Exaone4ForCausalLM": Exaone4ArchitectureAdapter,
    "FalconForCausalLM": FalconArchitectureAdapter,
    "FalconH1ForCausalLM": FalconH1ArchitectureAdapter,
    "FalconMambaForCausalLM": FalconMambaArchitectureAdapter,
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
    "GraniteForCausalLM": GraniteArchitectureAdapter,
    "GraniteMoeForCausalLM": GraniteMoeArchitectureAdapter,
    "GraniteMoeHybridForCausalLM": GraniteMoeHybridArchitectureAdapter,
    "GlmForCausalLM": GlmArchitectureAdapter,
    "GlmMoeDsaForCausalLM": GlmMoeDsaArchitectureAdapter,
    "Glm4MoeForCausalLM": Glm4MoeArchitectureAdapter,
    "Glm4MoeLiteForCausalLM": Glm4MoeLiteArchitectureAdapter,
    "GPT2LMHeadModel": GPT2ArchitectureAdapter,
    "GPTBigCodeForCausalLM": GPTBigCodeArchitectureAdapter,
    "GptOssForCausalLM": GPTOSSArchitectureAdapter,
    "GPT2LMHeadCustomModel": Gpt2LmHeadCustomArchitectureAdapter,
    "GPTJForCausalLM": GptjArchitectureAdapter,
    "HubertForCTC": HubertArchitectureAdapter,
    "HubertModel": HubertArchitectureAdapter,
    "HunYuanDenseV1ForCausalLM": HunYuanDenseV1ArchitectureAdapter,
    "Idefics3ForConditionalGeneration": Idefics3ArchitectureAdapter,
    "InternLM2ForCausalLM": InternLM2ArchitectureAdapter,
    "LlamaForCausalLM": LlamaArchitectureAdapter,
    "LlavaForConditionalGeneration": LlavaArchitectureAdapter,
    "LlavaNextForConditionalGeneration": LlavaNextArchitectureAdapter,
    "LlavaOnevisionForConditionalGeneration": LlavaOnevisionArchitectureAdapter,
    "Lfm2MoeForCausalLM": Lfm2MoeArchitectureAdapter,
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
    "MPTForCausalLM": MPTArchitectureAdapter,
    "NeoForCausalLM": NeoArchitectureAdapter,
    "NeoXForCausalLM": NeoxArchitectureAdapter,
    "NeelSoluOldForCausalLM": NeelSoluOldArchitectureAdapter,
    "OlmoForCausalLM": OlmoArchitectureAdapter,
    "Olmo2ForCausalLM": Olmo2ArchitectureAdapter,
    "Olmo3ForCausalLM": Olmo3ArchitectureAdapter,
    "OlmoeForCausalLM": OlmoeArchitectureAdapter,
    "OpenAIGPTLMHeadModel": OpenAIGPTArchitectureAdapter,
    "OpenELMForCausalLM": OpenElmArchitectureAdapter,
    "OPTForCausalLM": OptArchitectureAdapter,
    "PegasusForConditionalGeneration": PegasusArchitectureAdapter,
    "PhiForCausalLM": PhiArchitectureAdapter,
    "Phi3ForCausalLM": Phi3ArchitectureAdapter,
    "PhiMoEForCausalLM": PhiMoEArchitectureAdapter,
    "QwenForCausalLM": QwenArchitectureAdapter,
    "Qwen2ForCausalLM": Qwen2ArchitectureAdapter,
    "Qwen2AudioForConditionalGeneration": Qwen2AudioArchitectureAdapter,
    "Qwen2MoeForCausalLM": Qwen2MoeArchitectureAdapter,
    "Qwen3ForCausalLM": Qwen3ArchitectureAdapter,
    "Qwen3MoeForCausalLM": Qwen3MoeArchitectureAdapter,
    "Qwen3NextForCausalLM": Qwen3NextArchitectureAdapter,
    "Qwen3_5ForCausalLM": Qwen3_5ArchitectureAdapter,
    "Qwen3_5ForConditionalGeneration": Qwen3_5MultimodalArchitectureAdapter,
    "Qwen3_5MoeForCausalLM": Qwen3_5MoeArchitectureAdapter,
    "Qwen3_5MoeForConditionalGeneration": Qwen3_5MoeMultimodalArchitectureAdapter,
    "RecurrentGemmaForCausalLM": RecurrentGemmaArchitectureAdapter,
    "SeedOssForCausalLM": SeedOssArchitectureAdapter,
    "SmolLM3ForCausalLM": SmolLM3ArchitectureAdapter,
    "StableLmForCausalLM": StableLmArchitectureAdapter,
    "Starcoder2ForCausalLM": Starcoder2ArchitectureAdapter,
    "T5ForConditionalGeneration": T5ArchitectureAdapter,
    "MT5ForConditionalGeneration": T5ArchitectureAdapter,
    "T5WithLMHeadModel": T5ArchitectureAdapter,
    "T5GemmaForConditionalGeneration": T5GemmaArchitectureAdapter,
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
