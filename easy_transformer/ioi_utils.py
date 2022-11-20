from contextlib import suppress
import warnings
from functools import partial
from easy_transformer import EasyTransformer
import plotly.graph_objects as go
import numpy as np
from numpy import sin, cos, pi
from typing import List, Tuple, Dict, Union, Optional, Callable, Any
from tqdm import tqdm
import pandas as pd
import torch
import plotly.express as px
import gc
import einops
from easy_transformer.experiments import get_act_hook

from easy_transformer.ioi_dataset import IOIDataset
from easy_transformer.ioi_circuit_extraction import do_circuit_extraction

ALL_COLORS = px.colors.qualitative.Dark2
CLASS_COLORS = {
    "name mover": ALL_COLORS[0],
    "negative": ALL_COLORS[1],
    "s2 inhibition": ALL_COLORS[2],
    "induction": ALL_COLORS[5],
    "duplicate token": ALL_COLORS[3],
    "previous token": ALL_COLORS[6],
    "none": ALL_COLORS[7],
    "backup name mover": "rgb(27,100,119)",
    "light backup name mover": "rgb(146,183,210)",
}


from easy_transformer.ioi_circuit_extraction import get_extracted_idx

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


def max_2d(m, k=1):
    """Get the max of a matrix"""
    if len(m.shape) != 2:
        raise NotImplementedError()
    mf = m.flatten()
    inds = torch.topk(mf, k=k).indices
    out = []
    for ind in inds:
        ind = ind.item()
        x = ind // m.shape[1]
        y = ind - x * m.shape[1]
        out.append((x, y))
    return out, mf[inds]


def show_pp(
    m,
    xlabel="",
    ylabel="",
    title="",
    bartitle="",
    animate_axis=None,
    highlight_points=None,
    highlight_name="",
    return_fig=False,
    show_fig=True,
    **kwargs,
):
    """
    Plot a heatmap of the values in the matrix `m`
    """

    if animate_axis is None:
        fig = px.imshow(
            m,
            title=title if title else "",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            **kwargs,
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
        xaxis_range=[-0.5, m.shape[1] - 0.5],
        showlegend=True,
        legend=dict(x=-0.1),
    )
    if highlight_points is not None:
        fig.update_yaxes(range=[m.shape[0] - 0.5, -0.5], autorange=False)
    if show_fig:
        fig.show()
    if return_fig:
        return fig


# Plot attention patterns weighted by value norm


def show_attention_patterns(
    model,
    heads,
    ioi_dataset,
    precomputed_cache=None,
    mode="val",
    title_suffix="",
    return_fig=False,
    return_mtx=False,
):  # Arthur edited for one of my experiments, things work well
    assert mode in [
        "attn",
        "val",
        "scores",
    ]  # value weighted attention or attn for attention probas
    assert isinstance(
        ioi_dataset, IOIDataset
    ), f"ioi_dataset must be an IOIDataset {type(ioi_dataset)}"
    prompts = ioi_dataset.sentences
    assert len(heads) == 1 or not (return_fig or return_mtx)

    for (layer, head) in heads:
        cache = {}

        good_names = [
            f"blocks.{layer}.attn.hook_attn" + ("_scores" if mode == "scores" else "")
        ]
        if mode == "val":
            good_names.append(f"blocks.{layer}.attn.hook_v")
        if precomputed_cache is None:
            model.cache_some(
                cache=cache, names=lambda x: x in good_names
            )  # shape: batch head_no seq_len seq_len
            logits = model(ioi_dataset.toks.long())
        else:
            cache = precomputed_cache
        attn_results = torch.zeros(
            size=(ioi_dataset.N, ioi_dataset.max_len, ioi_dataset.max_len)
        )
        attn_results += -20

        for i, text in enumerate(prompts):
            # assert len(list(cache.items())) == 1 + int(mode == "val"), len(list(cache.items()))
            toks = ioi_dataset.toks[i]  # model.tokenizer(text)["input_ids"]
            current_length = len(toks)
            words = [model.tokenizer.decode([tok]) for tok in toks]
            attn = cache[good_names[0]].detach().cpu()[i, head, :, :]

            if mode == "val":
                vals = cache[good_names[1]].detach().cpu()[i, :, head, :].norm(dim=-1)
                cont = torch.einsum("ab,b->ab", attn, vals)

            fig = px.imshow(
                attn if mode in ["attn", "scores"] else cont,
                title=f"{layer}.{head} Attention" + title_suffix,
                color_continuous_midpoint=0,
                color_continuous_scale="RdBu",
                labels={"y": "Queries", "x": "Keys"},
                height=500,
            )

            fig.update_layout(
                xaxis={
                    "side": "top",
                    "ticktext": words,
                    "tickvals": list(range(len(words))),
                    "tickfont": dict(size=15),
                },
                yaxis={
                    "ticktext": words,
                    "tickvals": list(range(len(words))),
                    "tickfont": dict(size=15),
                },
            )
            if return_fig and not return_mtx:
                return fig
            elif return_mtx and not return_fig:
                attn_results[i, :current_length, :current_length] = (
                    attn[:current_length, :current_length].clone().cpu()
                )
            else:
                fig.show()

        if return_fig and not return_mtx:
            return fig
        elif return_mtx and not return_fig:
            return attn_results


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


global last_time
last_time = None
import time


def get_time(s):
    global last_time
    if last_time is None:
        last_time = time.time()
    else:
        print(f"Time elapsed - {s} -: {time.time() - last_time}")
        last_time = time.time()


def scatter_attention_and_contribution(
    model,
    layer_no,
    head_no,
    ioi_dataset,
    return_vals=False,
    return_fig=False,
):
    """
    Plot a scatter plot
    for each input sequence with the attention paid to IO and S
    and the amount that is written in the IO and S directions
    """

    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers
    model_unembed = model.unembed.W_U.detach().cpu()
    df = []
    cache = {}
    model.cache_all(cache)

    logits = model(ioi_dataset.toks.long())

    for i, prompt in enumerate(ioi_dataset.ioi_prompts):

        io_tok = model.tokenizer(" " + prompt["IO"])["input_ids"][0]
        s_tok = model.tokenizer(" " + prompt["S"])["input_ids"][0]
        toks = model.tokenizer(prompt["text"])["input_ids"]
        io_pos = toks.index(io_tok)
        s1_pos = toks.index(s_tok)
        s2_pos = toks[s1_pos + 1 :].index(s_tok) + (s1_pos + 1)
        assert toks[-1] == io_tok

        io_dir = model_unembed[:, io_tok].detach()
        s_dir = model_unembed[:, s_tok].detach()

        # model.reset_hooks() # should allow things to be done with ablated models

        for dire, posses, tok_type in [
            (io_dir, [io_pos], "IO"),
            (s_dir, [s1_pos, s2_pos], "S"),
        ]:
            prob = sum(
                [
                    cache[f"blocks.{layer_no}.attn.hook_attn"][
                        i, head_no, ioi_dataset.word_idx["end"][i], pos
                    ]
                    .detach()
                    .cpu()
                    for pos in posses
                ]
            )
            resid = (
                cache[f"blocks.{layer_no}.attn.hook_result"][
                    i, ioi_dataset.word_idx["end"][i], head_no, :
                ]
                .detach()
                .cpu()
            )
            dot = torch.einsum("a,a->", resid, dire)
            df.append([prob, dot, tok_type, prompt["text"]])

    # most of the pandas stuff is intuitive, no need to deeply understand
    viz_df = pd.DataFrame(
        df, columns=[f"Attn Prob on Name", f"Dot w Name Embed", "Name Type", "text"]
    )
    fig = px.scatter(
        viz_df,
        x=f"Attn Prob on Name",
        y=f"Dot w Name Embed",
        color="Name Type",
        hover_data=["text"],
        color_discrete_sequence=["rgb(114,255,100)", "rgb(201,165,247)"],
        title=f"How Strong {layer_no}.{head_no} Writes in the Name Embed Direction Relative to Attn Prob",
    )

    if return_vals:
        return viz_df
    if return_fig:
        return fig
    else:
        fig.show()


# metrics
# (Callable[ [EasyTransformer, IOIDataset], ...]) # probably a tensor, but with more stuff too as well sometimes


def handle_all_and_std(returning, all, std):
    """
    For use by the below functions. Lots of options!!!
    """

    if all and not std:
        return returning
    if std:
        if all:
            first_bit = (returning).detach().cpu()
        else:
            first_bit = (returning).mean().detach().cpu()
        return first_bit, torch.std(returning).detach().cpu()
    return (returning).mean().detach().cpu()


def logit_diff(
    model,
    ioi_dataset,
    all=False,
    std=False,
    both=False,
):  # changed by Arthur to take dataset object, :pray: no big backwards compatibility issues
    """
    Difference between the IO and the S logits at the "to" token
    """

    logits = model(ioi_dataset.toks.long()).detach()

    # uhhhh, I guess logit sum is constatn, but the constant is -516763 which seems weird (not 0?)
    # end_logits = logits[torch.arange(ioi_dataset.N), ioi_dataset.word_idx["end"], :]
    # assert len(end_logits.shape) == 2, end_logits.shape
    # assert torch.allclose(end_logits[0], end_logits[0] * 0.0)
    # for i in range(10):
    #     print(torch.sum(end_logits[i]))

    IO_logits = logits[
        torch.arange(len(ioi_dataset)),
        ioi_dataset.word_idx["end"],
        ioi_dataset.io_tokenIDs,
    ]
    S_logits = logits[
        torch.arange(len(ioi_dataset)),
        ioi_dataset.word_idx["end"],
        ioi_dataset.s_tokenIDs,
    ]

    if both:
        return handle_all_and_std(IO_logits, all, std), handle_all_and_std(
            S_logits, all, std
        )

    else:
        return handle_all_and_std(IO_logits - S_logits, all, std)


def attention_on_token(
    model, ioi_dataset, layer, head_idx, token, all=False, std=False, scores=False
):
    """
    Get the attention on token `token` from the end position
    """

    hook_name_raw = "blocks.{}.attn.hook_attn" + ("_scores" if scores else "")
    hook_name = hook_name_raw.format(layer)
    cache = {}
    model.cache_some(cache, lambda x: x == hook_name)
    # shape is batch * head * from * to
    logits = model(ioi_dataset.toks.long()).detach()
    atts = cache[hook_name][
        torch.arange(ioi_dataset.N),
        head_idx,
        ioi_dataset.word_idx["end"],
        ioi_dataset.word_idx[token],
    ]
    return handle_all_and_std(atts, all, std)


def positions(x: torch.Tensor):
    """
    x is a tensor of shape (B, L)
    returns the order of the elements in x
    """
    return torch.argsort(x, dim=1)


def posses(model, ioi_dataset, all=False, std=False):
    """
    Ranking of the IO token in all the tokens
    """
    text_prompts = ioi_dataset.text_prompts
    logits = model(text_prompts).detach().cpu()  # batch * sequence length * vocab_size
    warnings.warn("+1ing")
    end_logits = logits[
        torch.arange(len(text_prompts)), ioi_dataset.word_idx["end"] + 1, :
    ]  # batch * vocab_size

    positions = torch.argsort(end_logits, dim=1)
    io_positions = positions[torch.arange(len(text_prompts)), ioi_dataset.io_tokenIDs]

    return handle_all_and_std(io_positions, all, std)


def probs(model, ioi_dataset, all=False, std=False, type="io", verbose=False):
    """
    IO probs
    """

    logits = model(
        ioi_dataset.toks.long()
    ).detach()  # batch * sequence length * vocab_size
    end_logits = logits[
        torch.arange(len(ioi_dataset)), ioi_dataset.word_idx["end"], :
    ]  # batch * vocab_size

    end_probs = torch.softmax(end_logits, dim=1)

    if type == "io":
        token_ids = ioi_dataset.io_tokenIDs
    elif type == "s":
        token_ids = ioi_dataset.s_tokenIDs
    else:
        raise ValueError("type must be io or s")

    assert len(end_probs.shape) == 2
    io_probs = end_probs[torch.arange(ioi_dataset.N), token_ids]
    if verbose:
        print(io_probs)
    return handle_all_and_std(io_probs, all, std)


def get_top_tokens_and_probs(model, text_prompt):
    logits, tokens = model(
        text_prompt, prepend_bos=False, return_type="logits_and_tokens"
    )
    logits = logits.squeeze(0)
    end_probs = torch.softmax(logits, dim=1)
    # topk = torch.topk(end_probs[])
    return end_probs, tokens


def all_subsets(L: List) -> List[List]:
    """
    Returns all subsets of L
    """
    if len(L) == 0:
        return [[]]
    else:
        rest = all_subsets(L[1:])
        return rest + [[L[0]] + subset for subset in rest]  # thanks copilot


# some ellipse shit


def ellipse_arc(x_center=0, y_center=0, ax1=[1, 0], ax2=[0, 1], a=1, b=1, N=100):
    # x_center, y_center the coordinates of ellipse center
    # ax1 ax2 two orthonormal vectors representing the ellipse axis directions
    # a, b the ellipse parameters
    if abs(np.linalg.norm(ax1) - 1) > 1e-06 or abs(np.linalg.norm(ax2) - 1) > 1e-06:
        raise ValueError("ax1, ax2 must be unit vectors")
    if abs(np.dot(ax1, ax2)) > 1e-06:
        raise ValueError("ax1, ax2 must be orthogonal vectors")
    t = np.linspace(0, 2 * pi, N)
    # ellipse parameterization with respect to a system of axes of directions a1, a2
    xs = a * cos(t)
    ys = b * sin(t)
    # rotation matrix
    R = np.array([ax1, ax2]).T
    # coordinate of the  ellipse points with respect to the system of axes [1, 0], [0,1] with origin (0,0)
    xp, yp = np.dot(R, [xs, ys])
    x = xp + x_center
    y = yp + y_center
    return x, y


def ellipse_wht(mu, sigma):
    """
    Returns x, y and theta of confidence ellipse
    """
    vals, vecs = np.linalg.eigh(sigma)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.arctan2(*vecs[:, 0][::-1])  # grr copilot why degrees
    if theta < 0:
        theta += 2 * pi
    width, height = 2 * np.sqrt(vals)
    return width, height, theta


def plot_ellipse(fig, xs, ys, color="MediumPurple", nstd=1, name=""):
    mu = np.mean(xs), np.mean(ys)
    sigma = np.cov(xs, ys)
    w, h, t = ellipse_wht(mu, sigma)
    print(w, h, t)
    w *= nstd
    h *= nstd
    x, y = ellipse_arc(
        x_center=mu[0],
        y_center=mu[1],
        ax1=[cos(t), sin(t)],
        ax2=[-sin(t), cos(t)],
        a=w,
        b=h,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            marker=dict(size=20, color=color),
            name=name,
        )
    )


def get_heads_from_nodes(nodes, ioi_dataset):
    heads_to_keep_tok = {}
    for h, t in nodes:
        if h not in heads_to_keep_tok:
            heads_to_keep_tok[h] = []
        if t not in heads_to_keep_tok[h]:
            heads_to_keep_tok[h].append(t)

    heads_to_keep = {}
    for h in heads_to_keep_tok:
        heads_to_keep[h] = get_extracted_idx(heads_to_keep_tok[h], ioi_dataset)

    return heads_to_keep


def get_heads_from_nodes(nodes, ioi_dataset):
    heads_to_keep_tok = {}
    for h, t in nodes:
        if h not in heads_to_keep_tok:
            heads_to_keep_tok[h] = []
        if t not in heads_to_keep_tok[h]:
            heads_to_keep_tok[h].append(t)

    heads_to_keep = {}
    for h in heads_to_keep_tok:
        heads_to_keep[h] = get_extracted_idx(heads_to_keep_tok[h], ioi_dataset)

    return heads_to_keep


def circuit_from_nodes_logit_diff(model, ioi_dataset, nodes):
    """Take a list of nodes, return the logit diff of the circuit described by the nodes"""
    heads_to_keep = get_heads_from_nodes(nodes, ioi_dataset)
    # print(heads_to_keep)
    model.reset_hooks()
    model, _ = do_circuit_extraction(
        model=model,
        heads_to_keep=heads_to_keep,
        mlps_to_remove={},
        ioi_dataset=ioi_dataset,
    )
    return logit_diff(model, ioi_dataset, all=False)


def basis_change(x, y):
    """
    Change the basis (1, 0) and (0, 1) to the basis
    1/sqrt(2) (1, 1) and 1/sqrt(2) (-1, 1)
    """

    return (x + y) / np.sqrt(2), (y - x) / np.sqrt(2)


def add_arrow(fig, end_point, start_point, color="black"):
    x_start, y_start = start_point
    x_end, y_end = end_point
    fig.add_annotation(
        x=x_start,
        y=y_start,
        ax=x_end,
        ay=y_end,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text="",  # if you want only the arrow
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=color,
    )


def compute_next_tok_dot_prod(
    model,
    sentences,
    l,
    h,
    batch_size=1,
    seq_tokenized=False,
):
    """Compute dot product of model's next token logits with the logits of the next token in the sentences. Support batch_size > 1"""
    assert len(sentences) % batch_size == 0
    cache = {}
    model.cache_some(
        cache, lambda x: x in [f"blocks.{l}.attn.hook_result"], device="cuda"
    )
    if seq_tokenized:
        toks = sentences
    else:
        toks = model.tokenizer(sentences, padding=False).input_ids

    prod = []
    model_unembed = (
        model.unembed.W_U.detach().cpu()
    )  # note that for GPT2 embeddings and unembeddings are tided such that W_E = Transpose(W_U)
    for i in tqdm(range(len(sentences) // batch_size)):
        # get_time("pre forward")
        model.run_with_hooks(
            sentences[i * batch_size : (i + 1) * batch_size],
            reset_hooks_start=False,
            reset_hooks_end=False,
        )
        # get_time("post forward")
        # print_gpu_mem("post run")
        # get_time("pre prod")
        n_seq = len(sentences)
        for s in range(batch_size):
            idx = i * batch_size + s

            attn_result = cache[f"blocks.{l}.attn.hook_result"][
                s, : (len(toks[idx]) - 1), h, :
            ].cpu()  # nb seq, seq_len-1, embed dim
            next_tok = toks[idx][1:]  # nb_seq, seq_len-1

            next_tok_dir = model_unembed[next_tok]  # nb_seq, seq_len-1, dim
            # print(attn_result.shape, next_tok_dir.shape, len(toks[idx]) - 1)
            # print(next_tok_dir.shape, attn_result.shape)
            prod.append(
                torch.einsum("hd,hd->h", attn_result, next_tok_dir)
                .detach()
                .cpu()
                .numpy()
            )
        # get_time("post prod")
    # print_gpu_mem("post run")
    return prod


def get_gray_scale(val, min_val, max_val):
    max_col = 255
    min_col = 232
    max_val = max_val
    min_val = min_val
    val = val
    return int(min_col + ((max_col - min_col) / (max_val - min_val)) * (val - min_val))


def get_opacity(val, min_val, max_val):
    max_val = max_val
    min_val = min_val
    return (val - min_val) / (max_val - min_val)


def print_toks_with_color(toks, color, show_low=False, show_high=False, show_all=False):
    min_v = min(color)
    max_v = max(color)
    for i, t in enumerate(toks):
        c = get_gray_scale(color[i], min_v, max_v)
        text_c = 232 if c > 240 else 255
        show_value = show_all
        if show_low and c < 232 + 5:
            show_value = True
        if show_high and c > 255 - 5:
            show_value = True

        if show_value:
            if len(str(np.round(color[i], 2)).split(".")) > 1:
                val = (
                    str(np.round(color[i], 2)).split(".")[0]
                    + "."
                    + str(np.round(color[i], 2)).split(".")[1][:2]
                )
            else:
                val = str(np.round(color[i], 2))
            print(f"\033[48;5;{c}m\033[38;5;{text_c}m{t}({val})\033[0;0m", end="")
        else:
            print(f"\033[48;5;{c}m\033[38;5;{text_c}m{t}\033[0;0m", end="")


def tok_color_scale_to_html(toks, color):
    # print(len(toks), len(color))
    min_v = min(color)
    max_v = max(color)
    # display mix and max color in header
    html = (
        f'<span style="background-color: rgba({255},{0},{0}, {0})">Min: {min_v:.2f} </span>'
        + " "
        + f'<span style="background-color: rgba({255},{0},{0}, {255})">Max: {max_v:.2f}</span>'
        + "<br><br><br>"
    )
    for i, t in enumerate(toks):
        op = get_opacity(color[i], min_v, max_v)

        html += f'<span style="background-color: rgba({255},{0},{0}, {op})">{t}</span>'
    return html


def export_tok_col_to_file(folder, head, layer, tok_col, toks, chunck_name):
    if not os.path.isdir(folder):
        os.mkdir(folder)

    if not os.path.isdir(os.path.join(folder, f"layer_{layer}_head_{head}")):
        os.mkdir(os.path.join(folder, f"layer_{layer}_head_{head}"))

    filename = f"{folder}/layer_{layer}_head_{head}/layer_{layer}_head_{head}_{chunck_name}.html"
    all_html = ""
    for i in range(len(tok_col)):
        all_html += (
            f"<br><br><br>==============Sequence {i}=============<br><br><br>"
            + tok_color_scale_to_html(toks[i], tok_col[i])
        )
    with open(filename, "w") as f:
        f.write(all_html)


def find_owt_stimulus(
    model,
    owt_sentences,
    l,
    h,
    k=5,
    batch_size=1,
    export_to_html=False,
    folder="OWT_stimulus_by_head",
):
    prod = compute_next_tok_dot_prod(model, owt_sentences, l, h, batch_size=batch_size)

    min_prod = np.array([np.min(prod[i]) for i in range(len(prod))])
    max_prod = np.array([np.max(prod[i]) for i in range(len(prod))])

    # select 5 sequence with max and min prod values
    max_seq_idx = np.argsort(max_prod, axis=0)[-k:]
    min_seq_idx = np.argsort(min_prod, axis=0)[:k]

    # print(max_seq_idx)
    random_idx = np.random.choice(len(owt_sentences), k)

    max_seq = [
        show_tokens(owt_sentences[i], model, return_list=True) for i in max_seq_idx
    ]
    min_seq = [
        show_tokens(owt_sentences[i], model, return_list=True) for i in min_seq_idx
    ]
    random_seq = [
        show_tokens(owt_sentences[i], model, return_list=True) for i in random_idx
    ]
    max_seq_vals = [np.concatenate([np.array([0]), prod[i]]) for i in max_seq_idx]
    min_seq_vals = [np.concatenate([np.array([0]), prod[i]]) for i in min_seq_idx]
    random_seq_vals = [np.concatenate([np.array([0]), prod[i]]) for i in random_idx]

    if export_to_html:
        export_tok_col_to_file(
            folder,
            h,
            l,
            max_seq_vals,
            max_seq,
            "max",
        )
        export_tok_col_to_file(
            folder,
            h,
            l,
            min_seq_vals,
            min_seq,
            "min",
        )
        export_tok_col_to_file(
            folder,
            h,
            l,
            random_seq_vals,
            random_seq,
            "random",
        )
    else:

        print("\033[2;31;43m MAX ACTIVATION \033[0;0m")

        for seq_nb, s in enumerate(max_seq):
            # print(len(s), len(max_seq_vals[seq_nb]))
            print_toks_with_color(s, max_seq_vals[seq_nb], show_high=True)
            print("\n=========================\n")

        print("\033[2;31;43m Min ACTIVATION \033[0;0m")

        for seq_nb, s in enumerate(min_seq):
            print_toks_with_color(s, min_seq_vals[seq_nb], show_low=True)
            print("\n=========================\n")


#### Composition


def sample_activation(
    model, dataset: List[str], hook_names: List[str], n: int
) -> Dict[str, torch.Tensor]:
    data = np.random.choice(dataset, n)
    data = [str(elem) for elem in data]  # need to convert from numpy.str_
    cache = {}
    model.reset_hooks()
    model.cache_some(cache, lambda name: name in hook_names)
    _ = model(data)  # (batch, seq, vocab_size)
    model.reset_hooks()
    return cache


def get_head_param(model, module, layer, head):
    if module == "OV":
        W_v = model.blocks[layer].attn.W_V[head]
        W_o = model.blocks[layer].attn.W_O[head]
        W_ov = torch.einsum("hd,bh->db", W_v, W_o)
        return W_ov
    if module == "QK":
        W_k = model.blocks[layer].attn.W_K[head]
        W_q = model.blocks[layer].attn.W_Q[head]
        W_qk = torch.einsum("hd,hb->db", W_q, W_k)
        return W_qk
    if module == "Q":
        W_q = model.blocks[layer].attn.W_Q[head]
        return W_q
    if module == "K":
        W_k = model.blocks[layer].attn.W_K[head]
        return W_k
    if module == "V":
        W_v = model.blocks[layer].attn.W_V[head]
        return W_v
    if module == "O":
        W_o = model.blocks[layer].attn.W_O[head]
        return W_o
    raise ValueError(f"module {module} not supported")


def get_hook_name(model, module: str, layer: int, head: int) -> str:
    assert layer < model.cfg["n_layers"]
    assert head < model.cfg["n_heads"]
    if module == "OV" or module == "QK":
        return f"blocks.{layer}.hook_resid_pre"
    raise NotImplementedError("Module must be either OV or QK")


def compute_composition(
    model,
    dataset: List[str],
    n_samples: int,
    l1: int,
    h1: int,
    l2: int,
    h2: int,
    module_1: str,
    module_2: str,
):
    W_1 = get_head_param(model, module_1, l1, h1).detach()
    W_2 = get_head_param(model, module_2, l2, h2).detach()
    W_12 = torch.einsum("db,bc->dc", W_2, W_1)
    comp_scores = []

    baselines = []
    hook_name_1 = get_hook_name(module_1, l1, h1)
    hook_name_2 = get_hook_name(module_2, l2, h2)
    activations = sample_activation(
        model, dataset, [hook_name_1, hook_name_2], n_samples
    )
    # TODO: what to do with seq length dimension??
    # x_1 = activations[hook_name_1].mean(dim=1).squeeze().detach()
    # x_2 = activations[hook_name_2].mean(dim=1).squeeze().detach()
    x_1 = activations[hook_name_1].squeeze().detach()  # (batch, seq, d_model)
    x_2 = activations[hook_name_2].squeeze().detach()  # (batch, seq, d_model)

    # sanity check:
    # x_1 = torch.randn(768, device=W_1.device) / (768 ** 0.5)
    # x_2 = torch.randn(768, device=W_1.device) / (768 ** 0.5)
    c12 = torch.norm(torch.einsum("d e, b s e -> b s d", W_12, x_1), dim=-1)
    c1 = torch.norm(torch.einsum("d e, b s e -> b s d", W_1, x_1), dim=-1)
    c2 = torch.norm(torch.einsum("d e, b s e -> b s d", W_2, x_2), dim=-1)
    comp_score = c12 / (c1 * c2 * 768**0.5)
    comp_scores.append(comp_score)

    # compute baseline
    for _ in range(10):
        W_1b = torch.randn(W_1.shape, device=W_1.device) * W_1.std()
        W_2b = torch.randn(W_2.shape, device=W_2.device) * W_2.std()
        W_12b = torch.einsum("db,bc->dc", W_2b, W_1b)
        c12b = torch.norm(torch.einsum("d e, b s e -> b s d", W_12b, x_1), dim=-1)
        c1b = torch.norm(torch.einsum("d e, b s e -> b s d", W_1b, x_1), dim=-1)
        c2b = torch.norm(torch.einsum("d e, b s e -> b s d", W_2b, x_2), dim=-1)
        baseline = c12b / (c1b * c2b * 768**0.5)
        baselines.append(baseline)
    return (
        torch.stack(comp_scores).mean().cpu().numpy()
        - torch.stack(baselines).mean().cpu().numpy()
    )


def compute_composition_OV_QK(
    model,
    dataset: List[str],
    n_samples: int,
    l1: int,
    h1: int,
    l2: int,
    h2: int,
    mode: str,
):
    assert mode in ["Q", "K"]
    W_OV = get_head_param(model, "OV", l1, h1).detach()
    W_QK = get_head_param(model, "QK", l2, h2).detach()

    if mode == "Q":
        W_12 = torch.einsum("db,bc->dc", W_QK, W_OV)
    elif mode == "K":
        W_12 = torch.einsum("bc,bc->dc", W_OV, W_QK)  # OV^T * QK


def patch_all(z, source_act, hook):
    return source_act


def path_patching(
    model,
    D_new,
    D_orig,
    sender_heads,
    receiver_hooks,
    positions=["end"],
    return_hooks=False,
    extra_hooks=[],  # when we call reset hooks, we may want to add some extra hooks after this, add these here
    freeze_mlps=False,  # recall in IOI paper we consider these "vital model components"
    have_internal_interactions=False,
):
    """
    Patch in the effect of `sender_heads` on `receiver_hooks` only
    (though MLPs are "ignored" if `freeze_mlps` is False so are slight confounders in this case - see Appendix B of https://arxiv.org/pdf/2211.00593.pdf)

    TODO fix this: if max_layer < model.cfg.n_layers, then let some part of the model do computations (not frozen)
    """

    def patch_positions(z, source_act, hook, positions=["end"], verbose=False):
        for pos in positions:
            z[torch.arange(D_orig.N), D_orig.word_idx[pos]] = source_act[
                torch.arange(D_new.N), D_new.word_idx[pos]
            ]
        return z

    # process arguments
    sender_hooks = []
    for layer, head_idx in sender_heads:
        if head_idx is None:
            sender_hooks.append((f"blocks.{layer}.hook_mlp_out", None))

        else:
            sender_hooks.append((f"blocks.{layer}.attn.hook_result", head_idx))

    sender_hook_names = [x[0] for x in sender_hooks]
    receiver_hook_names = [x[0] for x in receiver_hooks]

    # Forward pass A (in https://arxiv.org/pdf/2211.00593.pdf)
    sender_cache = {}
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    model.cache_some(
        sender_cache, lambda x: x in sender_hook_names, suppress_warning=True
    )
    source_logits = model(D_new.toks.long())

    # Forward pass B
    target_cache = {}
    model.reset_hooks()
    for hook in extra_hooks:
        model.add_hook(*hook)
    model.cache_all(target_cache, suppress_warning=True)
    target_logits = model(D_orig.toks.long())

    # Forward pass C
    # Cache the receiver hooks
    # (adding these hooks first means we save values BEFORE they are overwritten)
    receiver_cache = {}
    model.reset_hooks()
    model.cache_some(
        receiver_cache,
        lambda x: x in receiver_hook_names,
        suppress_warning=True,
        verbose=False,
    )

    # "Freeze" intermediate heads to their D_orig values
    for layer in range(model.cfg.n_layers):
        for head_idx in range(model.cfg.n_heads):
            for hook_template in [
                "blocks.{}.attn.hook_q",
                "blocks.{}.attn.hook_k",
                "blocks.{}.attn.hook_v",
            ]:
                hook_name = hook_template.format(layer)

                if have_internal_interactions and hook_name in receiver_hook_names:
                    continue

                hook = get_act_hook(
                    patch_all,
                    alt_act=target_cache[hook_name],
                    idx=head_idx,
                    dim=2 if head_idx is not None else None,
                    name=hook_name,
                )
                model.add_hook(hook_name, hook)

        if freeze_mlps:
            hook_name = f"blocks.{layer}.hook_mlp_out"
            hook = get_act_hook(
                patch_all,
                alt_act=target_cache[hook_name],
                idx=None,
                dim=None,
                name=hook_name,
            )
            model.add_hook(hook_name, hook)

    for hook in extra_hooks:
        model.add_hook(*hook)

    # These hooks will overwrite the freezing, for the sender heads
    for hook_name, head_idx in sender_hooks:
        assert not torch.allclose(sender_cache[hook_name], target_cache[hook_name]), (
            hook_name,
            head_idx,
        )
        hook = get_act_hook(
            partial(patch_positions, positions=positions),
            alt_act=sender_cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        )
        model.add_hook(hook_name, hook)
    receiver_logits = model(D_orig.toks.long())

    # Add (or return) all the hooks needed for forward pass D
    model.reset_hooks()
    hooks = []
    for hook in extra_hooks:
        hooks.append(hook)

    for hook_name, head_idx in receiver_hooks:
        for pos in positions:
            if torch.allclose(
                receiver_cache[hook_name][torch.arange(D_orig.N), D_orig.word_idx[pos]],
                target_cache[hook_name][torch.arange(D_orig.N), D_orig.word_idx[pos]],
            ):
                warnings.warn("Torch all close for {}".format(hook_name))
        hook = get_act_hook(
            partial(patch_positions, positions=positions),
            alt_act=receiver_cache[hook_name],
            idx=head_idx,
            dim=2 if head_idx is not None else None,
            name=hook_name,
        )
        hooks.append((hook_name, hook))

    model.reset_hooks()
    if return_hooks:
        return hooks
    else:
        for hook_name, hook in hooks:
            model.add_hook(hook_name, hook)
        return model
