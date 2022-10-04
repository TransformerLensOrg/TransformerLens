from copy import deepcopy
from easy_transformer.experiments import (
    ExperimentMetric,
    AblationConfig,
    EasyAblation,
    EasyPatching,
    PatchingConfig,
    get_act_hook,
)

import torch
import plotly.express as px
import gc
import einops

from ioi_dataset import (
    IOIDataset,
    NOUNS_DICT,
    NAMES,
    gen_prompt_uniform,
    BABA_TEMPLATES,
    ABBA_TEMPLATES,
)


def list_diff(l1, l2):
    l2_ = [int(x) for x in l2]
    return list(set(l1).difference(set(l2_)))


def turn_keep_into_rmv(to_keep, max_len):
    to_rmv = {}
    for t in to_keep.keys():
        to_rmv[t] = []
        for idxs in to_keep[t]:
            to_rmv[t].append(list_diff(list(range(max_len)), idxs))
    return to_rmv


def process_heads_and_mlps(
    heads_to_remove=None,  # {(2,3) : List[List[int]]: dimensions dataset_size * datapoint_length
    mlps_to_remove=None,  # {2: List[List[int]]: dimensions dataset_size * datapoint_length
    heads_to_keep=None,  # as above for heads
    mlps_to_keep=None,  # as above for mlps
    ioi_dataset=None,
    model=None,
):
    assert (heads_to_remove is None) != (heads_to_keep is None)
    assert (mlps_to_keep is None) != (mlps_to_remove is None)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    dataset_length = len(ioi_dataset.text_prompts)

    if mlps_to_remove is not None:
        mlps = mlps_to_remove.copy()
    else:  # do smart computation in mean cache
        mlps = mlps_to_keep.copy()
        for l in range(n_layers):
            if l not in mlps_to_keep:
                mlps[l] = [[] for _ in range(dataset_length)]
        mlps = turn_keep_into_rmv(
            mlps, ioi_dataset.max_len
        )  # TODO check that this is still right for the max_len of maybe shortened datasets

    if heads_to_remove is not None:
        heads = heads_to_remove.copy()
    else:
        heads = heads_to_keep.copy()
        for l in range(n_layers):
            for h in range(n_heads):
                if (l, h) not in heads_to_keep:
                    heads[(l, h)] = [[] for _ in range(dataset_length)]
        heads = turn_keep_into_rmv(heads, ioi_dataset.max_len)
    return heads, mlps
    # print(mlps, heads)


def get_circuit_replacement_hook(
    heads_to_remove=None,
    mlps_to_remove=None,
    heads_to_keep=None,
    mlps_to_keep=None,
    heads_to_remove2=None,  # TODO @Alex ehat are these
    mlps_to_remove2=None,
    heads_to_keep2=None,
    mlps_to_keep2=None,
    ioi_dataset=None,
    model=None,
):
    heads, mlps = process_heads_and_mlps(
        heads_to_remove=heads_to_remove,  # {(2,3) : List[List[int]]: dimensions dataset_size * datapoint_length
        mlps_to_remove=mlps_to_remove,  # {2: List[List[int]]: dimensions dataset_size * datapoint_length
        heads_to_keep=heads_to_keep,  # as above for heads
        mlps_to_keep=mlps_to_keep,  # as above for mlps
        ioi_dataset=ioi_dataset,
        model=model,
    )

    if (heads_to_remove2 is not None) or (heads_to_keep2 is not None):
        heads2, mlps2 = process_heads_and_mlps(
            heads_to_remove=heads_to_remove2,  # {(2,3) : List[List[int]]: dimensions dataset_size * datapoint_length
            mlps_to_remove=mlps_to_remove2,  # {2: List[List[int]]: dimensions dataset_size * datapoint_length
            heads_to_keep=heads_to_keep2,  # as above for heads
            mlps_to_keep=mlps_to_keep2,  # as above for mlps
            ioi_dataset=ioi_dataset,
            model=model,
        )
    else:
        heads2, mlps2 = heads, mlps

    dataset_length = len(ioi_dataset.text_prompts)

    def circuit_replmt_hook(z, act, hook):  # batch, seq, heads, head dim
        layer = int(hook.name.split(".")[1])
        if "mlp" in hook.name and layer in mlps:
            for i in range(dataset_length):
                z[i, mlps[layer][i], :] = act[
                    i, mlps2[layer][i], :
                ]  # ablate all the indices in mlps[layer][i]; mean may contain semantic ablation
                # TODO can this i loop be vectorized?

        if "attn.hook_result" in hook.name and (layer, hook.ctx["idx"]) in heads:
            for i in range(dataset_length):  # we use the idx from contex to get the head
                z[i, heads[(layer, hook.ctx["idx"])][i], :] = act[
                    i,
                    heads2[(layer, hook.ctx["idx"])][i],
                    :,
                ]

        return z

    return circuit_replmt_hook, heads, mlps


def join_lists(l1, l2):  # l1 is a list of list. l2 a list of int. We add the int from l2 to the lists of l1.
    assert len(l1) == len(l2)
    assert type(l1[0]) == list and type(l2[0]) == int
    l = []
    for i in range(len(l1)):
        l.append(l1[i] + [l2[i]])
    return l


def get_extracted_idx(idx_list: list[str], ioi_dataset):
    int_idx = [[] for i in range(len(ioi_dataset.text_prompts))]
    for idx_name in idx_list:
        int_idx_to_add = [int(x) for x in list(ioi_dataset.word_idx[idx_name])]  # torch to python objects
        int_idx = join_lists(int_idx, int_idx_to_add)
    return int_idx


SMALL_CIRCUIT = {
    "name mover": [
        (9, 9),
        (10, 0),
        (9, 6),
    ],
    "negative": [],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (2, 9), (4, 11)],
}

MED_CIRCUIT = deepcopy(SMALL_CIRCUIT)
CIRCUIT = deepcopy(SMALL_CIRCUIT)

<<<<<<< HEAD
for head in [
    (9, 9),  # by importance
    (10, 0),
    (9, 6),
    (10, 10),
    (10, 6),
    (10, 2),
    (10, 1),
    (11, 2),
    (11, 9),
    (9, 7),
    (11, 3),
]:
=======
for head in [(10, 10), (10, 2), (11, 2), (10, 6), (10, 1), (11, 9), (9, 7), (11, 3), (11, 11)]:
>>>>>>> 43eeb1a3ec59b30098261fb8749d97b3b6911b29
    CIRCUIT["name mover"].append(head)

for head in [(10, 7), (11, 10)]:
    for circuit in [CIRCUIT, MED_CIRCUIT]:
        circuit["negative"].append(head)

CIRCUIT = {
        "name mover": [
            (9, 9),  # by importance
            (10, 0),
            (9, 6),
            (10, 10),
            (10, 6),
            (10, 2),
            (10, 1),
            (11, 2),
            (11, 9),
            (9, 7),
            (9, 0),
            (11, 11),
            (11, 3),
        ],
        "negative": [(10, 7), (11, 10)],
        "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
        "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
        "duplicate token": [(0, 1), (0, 10), (3, 0)],
        "previous token": [(2, 2), (2, 9), (4, 11)],
}

ARTHUR_CIRCUIT = deepcopy(CIRCUIT)
ARTHUR_CIRCUIT.pop("duplicate token")
ARTHUR_CIRCUIT["induction"] = [(5, 5), (6, 9)]

RELEVANT_TOKENS = {}
for head in CIRCUIT["name mover"] + CIRCUIT["negative"] + CIRCUIT["s2 inhibition"]:
    RELEVANT_TOKENS[head] = ["end"]

for head in CIRCUIT["induction"]:
    RELEVANT_TOKENS[head] = ["S2"]

for head in CIRCUIT["duplicate token"]:
    RELEVANT_TOKENS[head] = ["S2"]

for head in CIRCUIT["previous token"]:
    RELEVANT_TOKENS[head] = ["S+1"]

# ALEX_NAIVE_CIRCUIT = {
#     "name mover": [
#         (10, 6),
#         (9, 9),
#         (10, 2),
#     ],
#     "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
#     "duplicate token": [(1, 11), (0, 10), (3, 0)],
# }


ALEX_NAIVE = {
    "name mover": [(9, 6), (9, 9), (10, 0)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 9)],
    "duplicate token": [(3, 0), (0, 10)],
    "previous token": [(2, 2), (4, 11)],
    "negative": [],
}


def get_heads_circuit(ioi_dataset, excluded=[], mlp0=False, circuit=CIRCUIT):
    for excluded_thing in excluded:
        assert isinstance(excluded_thing, tuple) or excluded_thing in circuit.keys(), excluded_thing

    heads_to_keep = {}

    for circuit_class in circuit.keys():
        if circuit_class in excluded:
            continue
        for head in circuit[circuit_class]:
            if head in excluded:
                continue
            heads_to_keep[head] = get_extracted_idx(RELEVANT_TOKENS[head], ioi_dataset)

    if mlp0:
        mlps_to_keep = {}
        mlps_to_keep[0] = get_extracted_idx(
            ["IO", "and", "S", "S+1", "S2", "end"], ioi_dataset
        )  # IO, AND, S, S+1, S2, and END
        return heads_to_keep, mlps_to_keep

    return heads_to_keep


def do_circuit_extraction(
    heads_to_remove=None,  # {(2,3) : List[List[int]]: dimensions dataset_size * datapoint_length
    mlps_to_remove=None,  # {2: List[List[int]]: dimensions dataset_size * datapoint_length
    heads_to_keep=None,  # as above for heads
    mlps_to_keep=None,  # as above for mlps
    ioi_dataset=None,
    mean_dataset=None,
    model=None,
    metric=None,
    exclude_heads=[],
):
    """
    ..._to_remove means the indices ablated away. Otherwise the indices not ablated away.

    `exclude_heads` is a list of heads that actually we won't put any hooks on. Just keep them as is

    if `mean_dataset` is None, just use the ioi_dataset for mean
    """

    # check if we are either in keep XOR remove move from the args
    ablation, heads, mlps = get_circuit_replacement_hook(
        heads_to_remove=heads_to_remove,  # {(2,3) : List[List[int]]: dimensions dataset_size * datapoint_length
        mlps_to_remove=mlps_to_remove,  # {2: List[List[int]]: dimensions dataset_size * datapoint_length
        heads_to_keep=heads_to_keep,  # as above for heads
        mlps_to_keep=mlps_to_keep,  # as above for mlps
        ioi_dataset=ioi_dataset,
        model=model,
    )

    metric = ExperimentMetric(
        metric=metric, dataset=ioi_dataset.text_prompts, relative_metric=False
    )  # TODO make dummy metric

    if mean_dataset is None:
        mean_dataset = ioi_dataset
    config = AblationConfig(
        abl_type="custom",
        abl_fn=ablation,
        mean_dataset=mean_dataset.text_prompts,  # TODO nb of prompts useless ?
        target_module="attn_head",
        head_circuit="result",
        cache_means=True,  # circuit extraction *has* to cache means. the get_mean reset the
        verbose=True,
    )
    abl = EasyAblation(
        model,
        config,
        metric,
        semantic_indices=None,  # ioi_dataset.sem_tok_idx,
        mean_by_groups=True,  # TO CHECK CIRCUIT BY GROUPS
        groups=ioi_dataset.groups,
    )
    model.reset_hooks()

    for layer, head in heads.keys():
        if (layer, head) in exclude_heads:
            continue
        model.add_hook(*abl.get_hook(layer, head))
    for layer in mlps.keys():
        model.add_hook(*abl.get_hook(layer, head=None, target_module="mlp"))

    return model, abl


if __name__ == "__main__":
    print(CIRCUIT)
