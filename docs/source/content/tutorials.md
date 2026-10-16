# Tutorials

```{warning}
`HookedTransformer` and the other `Hooked*` classes were **removed in TransformerLens 4.0**. Use [`TransformerBridge`](migrating_to_v4.md) instead — call `enable_compatibility_mode()` after booting for HookedTransformer-equivalent numerics. See the [4.0 migration guide](migrating_to_v4.md).
```

- **Start with the [main demo](https://neelnanda.io/transformer-lens-demo) to learn how the library works, and the basic features**.

## Where To Start

- To see what using it for exploratory analysis in practice looks like, check out [my notebook analysing Indirect Object Identification](https://neelnanda.io/exploratory-analysis-demo) or [my recording of myself doing research](https://www.youtube.com/watch?v=yo4QvDn-vsU)!

- [What is a Transformer tutorial](https://neelnanda.io/transformer-tutorial)

## Demos

### Getting started

- [**Main Demo**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Main_Demo.ipynb) - The main demo. This is where to start if you're new to TransformerLens. Shows a lot of great features for getting started, including available models, how to access model activations, and generally useful features you should know about.

- [**Exploratory Analysis**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Exploratory_Analysis_Demo.ipynb) - Probably the best place to start, after the Main Demo. Demonstrates how to use TransformerLens to perform exploratory analysis - focuses less on rigor and more on getting a grasp of what's going on quickly. Uses a lot of useful interpretability techniques like logit attribution and activation patching. Steal liberally from this!

### Interpretability techniques

- [**Activation Patching in TransformerLens**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Activation_Patching_in_TL_Demo.ipynb) - Accompanies the [Exploratory Analysis Demo](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Exploratory_Analysis_Demo.ipynb). This demo explains how to use [Activation Patching](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=qeWBvs-R-taFfcCq-S_hgMqx) in TransformerLens, a mechanistic interpretability technique that uses causal intervention to identify which activations in a model matter for producing an output.

- [**Attribution Patching**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Attribution_Patching_Demo.ipynb) - [Attribution Patching](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) is an incomplete project that uses gradients to take a linear approximation to activation patching. It's a good approximation when patching in small activations like the outputs of individual attention heads, and bad when patching in large activations like a residual stream.

- [**Direct Logit Attribution**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Direct_Logit_Attribution_Demo.ipynb) - Direct Logit Attribution (DLA) measures how much each attention head, MLP neuron and embedding contributes to a target logit. This demo shows the `project_output_onto` argument of `ActivationCache.get_full_resid_decomposition`, which folds the projection into the decomposition so the per-neuron intermediate tensor, which can run to many GB, is never materialized.

- [**Direct Path Patching**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/direct_path_patching_ioi.ipynb) - Isolates a single head-to-head edge: patches only a source head's contribution into the query, key or value input of a destination head, leaving every other component's view of the source unchanged. Validated on the Indirect Object Identification (IOI) task with `get_act_patch_direct_path`.

- [**Head Detector**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Head_Detector_Demo.ipynb) - Shows how to use TransformerLens to automatically detect several common types of attention head, as well as create your own custom detection algorithms to find your own!

- [**Interactive Neuroscope**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb) - Very hacky demo, but this is a feature, not a bug. Shows how to quickly create useful web-based visualisations of data, even if you're not a professional front-end developer. This demo creates an interactive Neuroscope - a visualization of a neuron's activations on text that will dynamically update as you edit the text.

- [**Patchscopes & Generation with Patching**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Patchscopes_Generation_Demo.ipynb) - Implements [Patchscopes](https://arxiv.org/abs/2401.06102) (read a hidden representation from a source prompt and patch it into a target prompt), extends it to generating multiple tokens with patching, and applies it to logit lens, entity description and zero-shot feature extraction on GPT-2 small.

- [**SVD Interpreter Demo**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/SVD_Interpreter_Demo.ipynb) - Based on the [Conjecture post](https://www.lesswrong.com/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight#Directly_editing_SVD_representations) about how the singular value decompositions of transformer matrices are surprisingly interpretable, this demo shows how to use TransformerLens to reproduce this and investigate further.

The Backward Lens, Jacobian Lens and SVD Circuits notebooks are linked from [Choosing an analysis tool](analysis_tools.md) and [Fitting a Jacobian lens](jacobian_lens_fitting.md), and the SwiGLU Gate-Sign notebook from [The Hook System](hook_system.md).

### Models and architectures

- [**BERT**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/BERT.ipynb) - Uses BERT through `TransformerBridge` for masked language modelling and next sentence prediction, then inspects its activation cache and ablates an attention head with a hook.

- [**Gemma 3 Multimodal**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Gemma3_Multimodal.ipynb) - Loads the vision-language Gemma 3 (`google/gemma-3-4b-it`: a SigLIP vision encoder plus the Gemma 3 language model, the architecture MedGemma uses) through `TransformerBridge`, generates text from an image and a prompt, and captures vision-language activations with `run_with_cache()`. The checkpoint is gated on Hugging Face.

- [**GPT-OSS-20B**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/GPT_OSS_Demo.ipynb) - Loads OpenAI's [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) through `TransformerBridge` and walks through caching, expert-routing inspection, the logit lens, attention patterns and activation patching. Peak RAM is about 56 GB.

- [**LLaMA**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/LLaMA.ipynb) - Loads Meta's LLaMA models through `TransformerBridge`: either original LLaMA weights you have converted to the HuggingFace format, or the gated `meta-llama/Llama-3.2-1B` from the Hub. Compares logits with the HuggingFace model, then reads from and writes to hooks.

- [**LLaMA (Quantized)**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/LLaMA_GPU_Quantized.ipynb) - The LLaMA demo with `meta-llama/Llama-3.2-1B` loaded in 4-bit to reduce memory use (`bitsandbytes` on CUDA, `optimum-quanto` on MPS/CPU), plus per-head Q/K/V hooks via `use_split_qkv_input`.

- [**LLaVA**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/LLaVA.ipynb) - Loads LLaVA (`llava-hf/llava-1.5-7b-hf`: a CLIP vision encoder plus a LLaMA language model) through `TransformerBridge`, generates image descriptions, and extracts and visualizes attention patterns.

- [**Othello-GPT**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Othello_GPT.ipynb) - This is a demo notebook porting the weights of the Othello-GPT Model from the excellent [Emergent World Representations](https://arxiv.org/pdf/2210.13382.pdf) paper to TransformerLens. Neel's [sequence on investigating this](https://www.lesswrong.com/s/nhGNHyJHbrofpPbRG) is also well worth reading if you're interested in this topic!

- [**Qwen**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Qwen.ipynb) - Loads `Qwen/Qwen2-0.5B` through `TransformerBridge` and checks its output against the HuggingFace model. The first-generation Qwen section is commented out because that model's remote code needs `transformers_stream_generator`, which is incompatible with `transformers>=5.0.0`.

- [**StableLM**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/stable_lm.ipynb) - Compares Stability AI's base and instruction-tuned StableLM models (`stablelm-base-alpha-3b`, `stablelm-tuned-alpha-3b`) to show the effect of instruction fine-tuning, then explores a "Capture The Flag" prompt-injection attack on the tuned model.

- [**T5**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/T5.ipynb) - Loads `google-t5/t5-small` through `TransformerBridge`, runs it on token, string and list-of-string inputs, generates text, and visualizes encoder and decoder attention patterns and top-k tokens with CircuitsVis.

- [**Tracr to TransformerLens**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Tracr_to_Transformer_Lens_Demo.ipynb) - [Tracr](https://github.com/deepmind/tracr) is a cool new DeepMind tool that compiles a written program in [RASP](https://arxiv.org/abs/2106.06981) to transformer weights.This is a (hacky!) script to convert Tracr weights from the JAX form to a TransformerLens `TransformerBridge` in PyTorch.

### Backends and integrations

- [**HuggingFace-style Evals on TransformerBridge**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Bridge_Evals_Demo.ipynb) - Two ways to run benchmark evaluations on a `TransformerBridge` model: pass `bridge.original_model` to an existing framework such as `lm-eval-harness`, or write a short eval loop with hooks active to measure how an intervention (ablation, patching, steering) changes accuracy.

- [**Inspect Driver**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Inspect_Bridge_Demo.ipynb) - Uses `boot_inspect` to turn a model served through [`inspect_ai`](https://inspect.aisi.org.uk) into a bridge, so `run_with_cache`, named hook points and interventions work over the Inspect boundary. Runs on CPU.

- [**Inspect Driver (vLLM Provider)**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Inspect_vLLM_Provider_Demo.ipynb) - The Inspect driver demo served by vLLM (`provider="tl_bridge_vllm"`), so parallel `inspect_eval` samples stream through one resident model for dataset-scale evals. It serves fewer hook points than the HF provider and needs an NVIDIA GPU.

- [**LIT Integration**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/LIT_Integration_Demo.ipynb) - Uses Google's [Learning Interpretability Tool (LIT)](https://pair-code.github.io/lit/) with TransformerLens for interactive analysis: attention patterns, token predictions, the embedding projector and token salience.

### Training and research

- [**Grokking**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Grokking_Demo.ipynb) - "Grokking" is a phenomenon where a model can learn to memorise the training data (minimising training loss) but then, if trained for a lot longer, can learn to generalise, leading to a sharp decrease in test loss as well. This demo shows training a model on the task of modular addition, verifying that it groks, and doing analysis. The demo is light on explanation, so you'll probably want to pair it with [Neel's video series](https://www.youtube.com/watch?v=ob4vuiqG2Go) on the paper it's based on.

- [**No Position Experiment**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/No_Position_Experiment.ipynb) - The accompanying notebook to Neel's [real-time research video](https://www.youtube.com/watch?v=yo4QvDn-vsU). Trains a model with no positional embeddings to predict the previous token, and makes a start at analysing what's going on there!

- [**Real-time Training Telemetry**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Realtime_Training_Telemetry_Demo.ipynb) - Extracts and plots mechanistic metrics from the `ActivationCache` during a training loop to find the window where phase transitions, such as induction-head formation, begin. Trains a 2-layer toy model for 500 steps on CPU.
