# TransformerLens 3.0
**[Release Date]**

I am very excited to announce that TransformerLens 3.0 is here! This is the biggest release the project has ever shipped. 3.0 is a substantive architectural shift - one that we have been working toward for more than a year. The headline change is **TransformerBridge**, a new way of loading and instrumenting models that replaces the role `HookedTransformer.from_pretrained` used to play. This new system supports **48 architectures** and **9,000 models** out-of-the-box. Existing code continues to run through a compatibility layer, but new work should target the bridge, and over the next several minor releases we expect that to be the default path for all users.

If you have been following the dev-3.x branch, none of this will be a surprise. If you have been waiting on main, this release represents the accumulation of a great deal of work from a large number of contributors - thank you to everyone who helped make this happen.

## Lets start with some housekeeping

Before getting into what is new in this release, I want to take a moment to introduce myself for those of you who I have not yet interacted with. My name is Jonah Larson, I am a software engineer with a decade of professional experience developing enterprise software and other tools. Through my previous role, I'd worked with Bryce on several complex projects. He introduced me to TransformerLens and all of the cool things people have been getting up to with mechanistic interpretability. I have been following along with Bryce's work on 3.0 since summer 2025 and began contributing to TransformerLens in the fall of 2025. With Bryce needing to take a step back, I have stepped in as the primary maintainer with a focus on getting TransformerLens 3.0 shipped.

TransformerLens 3.0 would not have been possible without Bryce. He put an incredible amount of effort into the design of this new system. And he was an invaluable resource for me as I was finishing this release. If you enjoy this latest TransformerLens release, it is due in large part to him. Thank you Bryce for all your hard work.

Since I took over as primary maintainer in February 2026, I've been very pleased with the number of contributions we've been able to integrate into TransformerLens, and I appreciate each and every contributor who has dedicated time to resolving bugs and creating new model interfaces for TransformerLens 2.0. Those contributions are still reflected here in the legacy system, and have helped inform decisions about what we needed to address for this new version.

My primary goal in the coming month will be to address the remaining open PRs and resolve any bugs that are discovered in TransformerLens 3.0 as people begin to use it more widely. And of course, [my calendar](https://calendly.com/jonahtransformerlens/30min) is always open if you have ideas for new features, have a critical bug you need help with, or just want to chat about TransformerLens. I will also be putting out a user survey in the next couple weeks that will be open indefinitely as an easier outlet for providing feedback.

Additionally, for the next 3.1 release, I am working on an automated Claude tool for TransformerBridge architecture adapters. It will allow contributors to provide an architecture name and have the foundation for an architecture adapter created for them dynamically by Claude. This is the next big step for increasing TransformerLens's ability to support new models as they release, and just the first step towards my goal of potentially automating architecture adapter creation in the future.

## What changed

The single biggest pain point in TransformerLens 1.x and 2.x was adding new models. Every supported architecture had to be mapped into a single unified transformer implementation inside the library. This was a beautiful design in theory - interpretability code written once worked everywhere - but it meant that every new architecture required a significant implementation effort, and any divergence from the HuggingFace version was a latent source of bugs. In practice, new model support lagged behind the field, and the number of open issues asking for specific models kept growing.

TransformerBridge inverts this. Instead of reimplementing models, the bridge keeps the native HuggingFace implementation and wraps it behind a consistent interface through an **architecture adapter**. The adapter knows how the HF module graph maps onto a small set of generalized components - embedding, attention, MLP, normalization, blocks - and registers uniform hook points over them. The result is the same TransformerLens experience (hooks, caches, patching, generation) but applied to the real HF model, with one adapter per architecture instead of one implementation per model.

The numbers tell the story. At the time of the 2.0 release, TransformerLens supported roughly two dozen model families. TransformerBridge ships with support for **48 architectures**, spanning Llama, Mistral, Mixtral, Qwen, Gemma, OLMo, Phi, Granite, Cohere, DeepSeek, Mamba (state-space), LLaVA (multimodal), and many more. The full compatibility matrix - including which models have been verified end to end - is available on the new [TransformerBridge Models](../../generated/transformer_bridge_models.md) page.

For a complete migration guide with side-by-side examples, see [Migrating to TransformerLens 3](../migrating_to_v3.md).

## Breaking changes and deprecations

Two deprecations announced in 2.0 have now been removed - the `move_model` parameter in `ActivationCache.to` and the `cache_all` function in `hook_points`. If you are upgrading from 2.x and you were still relying on either, you will need to update your code before this release will work for you.

Beyond the 2.0 deprecations, there are a handful of new behaviors in 3.0 worth understanding:

### Weight processing is now opt-in

`HookedTransformer.from_pretrained` applied `fold_ln`, `center_writing_weights`, and `center_unembed` by default when loading a model. The bridge does **not** apply any of these on load - the raw HF weights are preserved. If your work depends on folded and centered weights, call `enable_compatibility_mode()` after booting the bridge to get the same processing the old defaults gave you. The migration guide walks through this in detail.

### Hook names are moving to a uniform convention

The canonical hook names on the bridge use `hook_in` / `hook_out` uniformly across components. For example, `blocks.0.hook_resid_pre` is now `blocks.0.hook_in` in canonical form. The old names still work through an alias layer, so existing notebooks and tests continue to run unchanged. New code should prefer the canonical names - the [Model Structure](../model_structure.md) page has the full mapping.

### Model name aliases are deprecated

`HookedTransformer.from_pretrained` accepted a wide range of short aliases like `"gpt2"`, `"llama-7b-hf"`, and `"gpt-neo-125M"`. The bridge also accepts these aliases for backwards compatibility, but it will emit a deprecation warning and prefer the full HuggingFace model id (e.g. `"openai-community/gpt2"`). The legacy aliases will be removed in the next major version.

### Load-time parameters that were removed

A few parameters that existed on `HookedTransformer.from_pretrained` are not part of `boot_transformers`: `n_devices`, `move_to_device`, `first_n_layers`, and `n_ctx`. If you were relying on any of these, please open an issue describing your use case so we can design the right pattern under the bridge.

## Roadmap

As Bryce did in his 2.0 announcement, I broke this into three timeframes.

### Immediate - within the next month

The primary focus right after 3.0 ships is smoothing out rough edges. Expect rapid 3.x patch releases to address bugs as they are reported. I would especially appreciate feedback from anyone migrating a large existing project - if something used to work on 2.x and does not work on 3.x (even through the compatibility layer), that is a bug and I want to know about it.

### Mid-term - within the next 3 months

The below is a draft. Priorities will shift based on user feedback.

#### Expanding architecture coverage

Even with 48 architectures supported, there are still a handful of frequently-requested models that do not yet have a bridge adapter. The goal is to close the most visible gaps and to document the adapter-writing process well enough that community contributors can add new architectures without needing deep familiarity with the bridge internals.

#### Multi-device and quantized loading under the bridge

The `n_devices` and `move_to_device` options from `HookedTransformer.from_pretrained` are not currently supported by the bridge. The right pattern here is still being worked out - the bridge wraps an HF model, so some of this naturally delegates to HuggingFace's own device and quantization machinery, but we need to document and test the end-to-end flow. Expect this to land as a mid-term deliverable.

### Long-term - within the next year

#### Deprecating HookedTransformer

The long-term goal is to make the bridge the single supported path for loading models. `HookedTransformer` will continue to work through 3.x, but we intend to remove it in the next major version. We will keep the community abreast of this decision if and when it is made officially.

#### Support for SSM and Hybrid models

The initial launch of TransformerLens 3.0 includes preliminary support for several hybrid architectures and two Mamba architectures with SSM. These are rough, proof-of-concept implementations that will need additional work for full compatibility.

#### Better tooling for adapter authors

Adding a new architecture today means writing an adapter and running the bridge's verification suite. That process works, but it can be smoothed significantly - better error messages when a mapping is wrong, a scaffold command that generates the adapter skeleton from an HF config, and a more thorough dashboard for diagnosing why a newly-added model's outputs don't match HuggingFace. This is also where the automated Claude tool will come into play. Development on this has already begun.

## Contributors

This next section is only relevant to contributors, so if anyone is reading this who is only using TransformerLens as a tool, then you can skip this section.

### Move from Poetry to UV

As part of the 3.0 cycle we switched the project's package and environment management from [Poetry](https://python-poetry.org/) to [UV](https://docs.astral.sh/uv/). UV is significantly faster for resolves and installs, handles lock files more predictably, and integrates cleanly with the scripts we use for building and hot-reloading the docs. If you have an existing Poetry-based checkout, delete your old `.venv`, install UV, and run `uv sync` to reset your environment. The updated setup instructions live on the [Contributing](../contributing.md) page.

### Dev branch changes

During the 3.x development cycle we maintained a `dev-3.x` branch for bridge work alongside the regular `dev` branch. With 3.0 shipping, `dev-3.x` has been merged into `main`, and the regular `dev` branch is now the active development branch again. New pull requests should target `dev`.

### Test suite changes

The bridge comes with a significant new verification suite that checks the outputs of each bridge adapter against the underlying HuggingFace model. The results feed the [TransformerBridge Models](../../generated/transformer_bridge_models.md) compatibility matrix. Anyone adding a new adapter will go through this suite as part of the PR process.

## Conclusion

Thank you for reading. TransformerLens 3.0 is the most substantial release this project has ever shipped, and it is the culmination of more than a year of design work, implementation, and iteration. My hope is that by keeping the native HuggingFace implementation and layering hooks on top through adapters, we can finally close the gap between "models TransformerLens supports" and "models people want to study," and keep that gap closed as the field continues to accelerate.

If you hit a bug, a missing feature, or a rough edge while migrating, please open an issue. And as before, if you are using TransformerLens and would like to chat about how it is serving you, I would love to hear from you.

## Appendix

### What "3.0" means under Semantic Versioning

Following the conventions laid out in the 2.0 announcement: the jump to 3.0 is required because we are removing pieces of the public API (specifically, the deprecations from 2.0 and a small number of `HookedTransformer.from_pretrained` load-time parameters). Everything else is either additive - the bridge, new adapters, new hook points - or preserved through the compatibility and alias layers. Code you write against the 3.0 API will continue to work unchanged through the 3.x branch.

### A note on the compatibility layer

The alias layer that maps old hook names to canonical bridge names is a deliberate choice, not a permanent fixture. It exists to make migration easy, and I expect it to stay around through all of 3.x. But aliases have a real maintenance cost, and when we eventually plan 4.0, removing the alias layer is one of the candidate changes. New code should use the canonical names so that a future migration is a non-event.
