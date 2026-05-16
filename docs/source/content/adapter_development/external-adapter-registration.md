# External Architecture Adapter Registration

TransformerLens supports loading custom architecture adapters from **external packages** — no fork required. You can write your adapter, register it, and use it with `boot_transformers()` without modifying TransformerLens source code.

## Two ways to register

### 1. Runtime registration

Call `register_adapter()` in your startup code:

```python
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)

ArchitectureAdapterFactory.register_adapter(
    "MyModelForCausalLM",
    MyArchitectureAdapter,
)

# Now this works:
bridge = TransformerBridge.boot_transformers("my-org/my-model")
```

> **Important:** The architecture name you register (e.g. `"MyModelForCausalLM"`) must match the `architectures` field in the model's HuggingFace `config.json`. TransformerLens reads this field to look up the adapter.

### 2. Entry-point registration (recommended for packages)

Declare your adapter in your package's `pyproject.toml`:

```toml
[project.entry-points."transformer_lens.architectures"]
"MyModelForCausalLM" = "my_package.adapters:MyArchitectureAdapter"
```

TransformerLens discovers these automatically on first use. Users just install your package alongside TransformerLens and `boot_transformers()` finds it.

## How it works

When `boot_transformers()` is called:

1. It reads the model's HuggingFace `config.json` to extract the `architectures` field. This field lists the architecture class name (e.g. `"MyModelForCausalLM"`). **This is the name you must use in your registration.**
2. `select_architecture_adapter()` checks the registry for that architecture name.
3. On first call, `discover_entry_points()` scans all installed packages for adapters declared via the `transformer_lens.architectures` entry-point group.
4. The matching adapter class is instantiated and used to build the bridge.

## Writing an adapter

Follow the [Architecture Adapter Creation Guide](adapter-creation-guide.md) to build your adapter class. Once written, use either registration method above to plug it into TransformerLens.

## Example package layout

```
my_transformer_plugin/
├── pyproject.toml          # declares the entry point
└── my_transformer_plugin/
    ├── __init__.py
    └── adapters.py         # contains MyArchitectureAdapter
```

**pyproject.toml:**

```toml
[project]
name = "my-transformer-plugin"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["transformer-lens>=3.0"]

[project.entry-points."transformer_lens.architectures"]
"MyModelForCausalLM" = "my_transformer_plugin.adapters:MyArchitectureAdapter"
```

**adapters.py:**

```python
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    # ... import the bridge components you need
)

class MyArchitectureAdapter(ArchitectureAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Set config, weight processing, component mapping
        # See the Adapter Creation Guide for details
```
