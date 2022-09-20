import torch
import plotly.express as px
import gc
import einops

from interp.circuit.projects.ioi.ioi_methods import N_LAYER

# other utils


def clear_gpu_mem():
    gc.collect()
    torch.cuda.empty_cache()


def show_tokens(tokens, model, return_list=False):
    # Prints the tokens as text, separated by |
    if type(tokens) == str:
        # If we input text, tokenize first
        tokens = model.to_tokens(tokens)
    text_tokens = [model.tokenizer.decode(t) for t in tokens.squeeze()]
    if return_list:
        return text_tokens
    else:
        print("|".join(text_tokens))


def show_pp(
    m,
    xlabel="",
    ylabel="",
    title="",
    bartitle="",
    animate_axis=None,
    highlight_points=None,
    highlight_name="",
    **kwargs,
):
    """
    Plot a heatmap of the values in the matrix `m`
    """

    if animate_axis is None:
        fig = px.imshow(
            m.T, title=title if title else "", color_continuous_scale="RdBu", color_continuous_midpoint=0, **kwargs
        )

    else:
        fig = px.imshow(
            einops.rearrange(m, "a b c -> a c b"),
            title=title if title else "",
            animation_frame=animate_axis,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            **kwargs,
        )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title=bartitle,
            thicknessmode="pixels",
            thickness=50,
            lenmode="pixels",
            len=300,
            yanchor="top",
            y=1,
            ticks="outside",
        ),
        xaxis_title="",
    )

    if highlight_points is not None:
        fig.add_scatter(
            x=highlight_points[1],
            y=highlight_points[0],
            mode="markers",
            marker=dict(color="green", size=10, opacity=0.5),
            name=highlight_name,
        )

    fig.update_layout(
        yaxis_title=ylabel,
        xaxis_title=xlabel,
        xaxis_range=[-0.5, m.T.shape[0] - 0.5],
        showlegend=True,
        legend=dict(x=-0.1),
    )
    fig.update_yaxes(range=[m.T.shape[1] - 0.5, -0.5], autorange=False)
    fig.show()


# Plot attention patterns weighted by value norm


def show_attention_patterns(model, heads, texts, mode="val", title_suffix=""):
    assert mode in [
        "attn",
        "val",
    ]  # value weighted attention or attn for attention probas
    assert type(texts) == list

    for (layer, head) in heads:
        cache = {}

        good_names = [f"blocks.{layer}.attn.hook_attn"]
        if mode == "val":
            good_names.append(f"blocks.{layer}.attn.hook_v")
        model.cache_some(cache=cache, names=lambda x: x in good_names)  # shape: batch head_no seq_len seq_len

        logits = model(texts)

        for i, text in enumerate(texts):
            assert len(list(cache.items())) == 1 + int(mode == "val"), len(list(cache.items()))
            toks = model.tokenizer(text)["input_ids"]
            words = [model.tokenizer.decode([tok]) for tok in toks]
            attn = cache[good_names[0]].detach().cpu()[i, head, :, :]
            if mode == "val":
                vals = cache[good_names[1]].detach().cpu()[i, :, head, :].norm(dim=-1)
                cont = torch.einsum("ab,b->ab", attn, vals)

            fig = px.imshow(
                attn if mode == "attn" else cont,
                title=f"{layer}.{head} Attention" + title_suffix,
                color_continuous_midpoint=0,
                color_continuous_scale="RdBu",
                labels={"y": "Queries", "x": "Keys"},
            )

            fig.update_layout(
                xaxis={
                    "side": "top",
                    "ticktext": words,
                    "tickvals": list(range(len(words))),
                    "tickfont": dict(size=8),
                },
                yaxis={
                    "ticktext": words,
                    "tickvals": list(range(len(words))),
                    "tickfont": dict(size=8),
                },
            )

            fig.show()


def safe_del(a):
    """Try and delete a even if it doesn't yet exist"""
    try:
        exec(f"del {a}")
    except:
        pass
    torch.cuda.empty_cache()


def get_indices_from_sql_file(fname, trial_id):
    """
    Given a SQL file, return the indices of the trial_id
    """
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(fname)
    df = pd.read_sql_query("SELECT * from trial_params", conn)
    return list(map(int, df[df.trial_id == trial_id].param_value.values))


if __name__ == "__main__":
    inds = get_indices_from_sql_file("example-study.db", 1494)
    print(inds)
