# %%
from functools import lru_cache

from easy_transformer import loading

# %%
# cfg = (loading.get_pretrained_model_config("solu-1l"))
# print(cfg)
# %%
""" 
Structure:
d_model, d_mlp, d_head, d_vocab, act_fn, n_heads, n_layers, n_ctx, n_params, 
Make an architecture table separately probs
tokenizer_name, training_data, has checkpoints
act_fn includes attn_only
architecture
Architecture should list weird shit to be aware of.
"""
import pandas as pd


# df = pd.DataFrame(np.random.randn(2, 2))
# print(df.to_markdown(open("test.md", "w")))
# %%
@lru_cache(maxsize=None)
def get_config(model_name):
    return loading.get_pretrained_model_config(model_name)


def get_property(name, model_name):
    cfg = get_config(model_name)
    if name == "act_fn":
        if cfg.attn_only:
            return "attn_only"
        elif cfg.act_fn == "gelu_new":
            return "gelu"
        elif cfg.act_fn == "gelu_fast":
            return "gelu"
        elif cfg.act_fn == "solu_ln":
            return "solu"
        else:
            return cfg.act_fn
    if name == "n_params":
        n_params = cfg.n_params
        if n_params < 1e4:
            return f"{n_params/1e3:.1f}K"
        elif n_params < 1e6:
            return f"{round(n_params/1e3)}K"
        elif n_params < 1e7:
            return f"{n_params/1e6:.1f}M"
        elif n_params < 1e9:
            return f"{round(n_params/1e6)}M"
        elif n_params < 1e10:
            return f"{n_params/1e9:.1f}B"
        elif n_params < 1e12:
            return f"{round(n_params/1e9)}B"
        else:
            raise ValueError(f"Passed in {n_params} above 1T?")
    else:
        return cfg.to_dict()[name]


if __name__ == "__main__":
    column_names = "n_params, n_layers, d_model, n_heads, act_fn, n_ctx, d_vocab, d_head, d_mlp".split(
        ", "
    )
    df = pd.DataFrame(
        {
            name: [
                get_property(name, model_name)
                for model_name in loading.DEFAULT_MODEL_ALIASES
            ]
            for name in column_names
        },
        index=loading.DEFAULT_MODEL_ALIASES,
    )
    df.to_markdown(open("model_properties_table.md", "w"))
