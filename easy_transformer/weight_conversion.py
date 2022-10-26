from easy_transformer import EasyTransformerConfig
import einops
import torch

VALID_PRETRAINED_MODEL_NAMES = set(
    [
        "gpt2", # Alias for GPT-2 Small
        "gpt2-small",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "distilgpt2",
        "facebook/opt-125m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b",
        "facebook/opt-13b",
        "facebook/opt-30b",
        "facebook/opt-66b",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-1.3B",
        "EleutherAI/gpt-neo-2.7B",
        "stanford-gpt2-small-A",
        "stanford-gpt2-small-B",
        "stanford-gpt2-small-C",
        "stanford-gpt2-small-D",
        "stanford-gpt2-small-E",
        "stanford-gpt2-medium-A",
        "stanford-gpt2-medium-B",
        "stanford-gpt2-medium-C",
        "stanford-gpt2-medium-D",
        "stanford-gpt2-medium-E",
        "solu-1l-old",
        "solu-2l-old",
        "solu-4l-old",
        "solu-6l-old",
        "solu-8l-old",
        "solu-10l-old",
        "solu-1l-c4-code",
        "solu-2l-c4-code",
        "solu-3l-c4-code",
        "solu-4l-c4-code",
        "attn-only-2l-induction-demo",
        "EleutherAI/gpt-j-6B",
        "EleutherAI/gpt-neox-20b",
        "EleutherAI/pythia-19m",
        "EleutherAI/pythia-125m",
        "EleutherAI/pythia-350m",
        "EleutherAI/pythia-800m",
        "EleutherAI/pythia-1.3b",
        "EleutherAI/pythia-6.7b",
        "EleutherAI/pythia-13b",
        "EleutherAI/pythia-125m-deduped",
        "EleutherAI/pythia-800m-deduped",
        "EleutherAI/pythia-1.3b-deduped",
        "EleutherAI/pythia-6.7b-deduped",
        "EleutherAI/pythia-13b-deduped",
    ]
)

# Maps things to their official HuggingFace name
PRETRAINED_MODEL_NAMES_DICT = {
    "stanford-gpt2-small-A": "stanford-crfm/alias-gpt2-small-x21",
    "stanford-gpt2-small-B": "stanford-crfm/battlestar-gpt2-small-x49",
    "stanford-gpt2-small-C": "stanford-crfm/caprica-gpt2-small-x81",
    "stanford-gpt2-small-D": "stanford-crfm/darkmatter-gpt2-small-x343",
    "stanford-gpt2-small-E": "stanford-crfm/expanse-gpt2-small-x777",
    "stanford-gpt2-medium-A": "stanford-crfm/arwen-gpt2-medium-x21",
    "stanford-gpt2-medium-B": "stanford-crfm/beren-gpt2-medium-x49",
    "stanford-gpt2-medium-C": "stanford-crfm/celebrimbor-gpt2-medium-x81",
    "stanford-gpt2-medium-D": "stanford-crfm/durin-gpt2-medium-x343",
    "stanford-gpt2-medium-E": "stanford-crfm/eowyn-gpt2-medium-x777",
    "gpt2-small": "gpt2",
}
# The steps for which there are checkpoints in the stanford crfm models - provided as reference
STANFORD_CRFM_CHECKPOINTS = (
    list(range(0, 100, 10))
    + list(range(100, 2000, 50))
    + list(range(2000, 20000, 100))
    + list(range(20000, 400000 + 1, 1000))
)

def convert_gpt2_weights(gpt2, cfg: EasyTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = gpt2.transformer.wte.weight
    state_dict["pos_embed.W_pos"] = gpt2.transformer.wpe.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = gpt2.transformer.h[l].ln_1.weight
        state_dict[f"blocks.{l}.ln1.b"] = gpt2.transformer.h[l].ln_1.bias
        
        # In GPT-2, q,k,v are produced by one big linear map, whose output is
        # concat([q, k, v])
        W = gpt2.transformer.h[l].attn.c_attn.weight
        W_Q, W_K, W_V = torch.tensor_split(W, 3, dim=1)
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "m (i h)->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "m (i h)->i m h", i=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        qkv_bias = gpt2.transformer.h[l].attn.c_attn.bias
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0]
        state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1]
        state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2]

        W_O = gpt2.transformer.h[l].attn.c_proj.weight
        W_O = einops.rearrange(W_O, "(i h) m->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = gpt2.transformer.h[l].attn.c_proj.bias

        
        state_dict[f"blocks.{l}.ln2.w"] = gpt2.transformer.h[l].ln_2.weight
        state_dict[f"blocks.{l}.ln2.b"] = gpt2.transformer.h[l].ln_2.bias
        
        W_in = gpt2.transformer.h[l].mlp.c_fc.weight
        state_dict[f"blocks.{l}.mlp.W_in"] = W_in
        state_dict[f"blocks.{l}.mlp.b_in"] = gpt2.transformer.h[l].mlp.c_fc.bias
        
        W_out = gpt2.transformer.h[l].mlp.c_proj.weight
        state_dict[f"blocks.{l}.mlp.W_out"] = W_out
        state_dict[f"blocks.{l}.mlp.b_out"] = gpt2.transformer.h[l].mlp.c_proj.bias
    state_dict[f"unembed.W_U"] = gpt2.lm_head.weight.T
    
    state_dict["ln_final.w"] = gpt2.transformer.ln_f.weight
    state_dict["ln_final.b"] = gpt2.transformer.ln_f.bias
    return state_dict

def convert_neo_weights(neo, cfg: EasyTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = neo.transformer.wte.weight
    state_dict["pos_embed.W_pos"] = neo.transformer.wpe.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = neo.transformer.h[l].ln_1.weight
        state_dict[f"blocks.{l}.ln1.b"] = neo.transformer.h[l].ln_1.bias
        
        W_Q = neo.transformer.h[l].attn.attention.q_proj.weight
        W_K = neo.transformer.h[l].attn.attention.k_proj.weight
        W_V = neo.transformer.h[l].attn.attention.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(cfg.n_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(cfg.n_heads, cfg.d_head)

        W_O = neo.transformer.h[l].attn.attention.out_proj.weight
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = neo.transformer.h[
            l
        ].attn.attention.out_proj.bias

        state_dict[f"blocks.{l}.ln2.w"] = neo.transformer.h[l].ln_2.weight
        state_dict[f"blocks.{l}.ln2.b"] = neo.transformer.h[l].ln_2.bias
        
        state_dict[f"blocks.{l}.mlp.W_in"] = neo.transformer.h[l].mlp.c_fc.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = neo.transformer.h[l].mlp.c_fc.bias
        
        state_dict[f"blocks.{l}.mlp.W_out"] = neo.transformer.h[l].mlp.c_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = neo.transformer.h[l].mlp.c_proj.bias
    state_dict["ln_final.w"] = neo.transformer.ln_f.weight
    state_dict["ln_final.b"] = neo.transformer.ln_f.bias
    
    state_dict["unembed.W_U"] = neo.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab)
    return state_dict

def convert_gptj_weights(gptj, cfg: EasyTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = gptj.transformer.wte.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = gptj.transformer.h[l].ln_1.weight
        state_dict[f"blocks.{l}.ln1.b"] = gptj.transformer.h[l].ln_1.bias
        
        W_Q = gptj.transformer.h[l].attn.q_proj.weight
        W_K = gptj.transformer.h[l].attn.k_proj.weight
        W_V = gptj.transformer.h[l].attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(i h) m->i m h", i=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(i h) m->i m h", i=cfg.n_heads)
        W_V = einops.rearrange(W_V, "(i h) m->i m h", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_K"] = torch.zeros(cfg.n_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_V"] = torch.zeros(cfg.n_heads, cfg.d_head)

        W_O = gptj.transformer.h[l].attn.out_proj.weight
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model)
        
        # Layer Norm 1 and 2 are tied.
        state_dict[f"blocks.{l}.ln2.w"] = state_dict[f"blocks.{l}.ln1.w"]
        state_dict[f"blocks.{l}.ln2.b"] = state_dict[f"blocks.{l}.ln1.b"]

        state_dict[f"blocks.{l}.mlp.W_in"] = gptj.transformer.h[l].mlp.fc_in.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = gptj.transformer.h[l].mlp.fc_in.bias
        
        state_dict[f"blocks.{l}.mlp.W_out"] = gptj.transformer.h[l].mlp.fc_out.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = gptj.transformer.h[l].mlp.fc_out.bias
    state_dict["ln_final.w"] = gptj.transformer.ln_f.weight
    state_dict["ln_final.b"] = gptj.transformer.ln_f.bias
    
    state_dict["unembed.W_U"] = gptj.lm_head.weight.T
    # Contains a bias, for some reason?
    state_dict["unembed.b_U"] = gptj.lm_head.bias
    return state_dict

def convert_neox_weights(neox, cfg: EasyTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = neox.gpt_neox.embed_in.weight

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = neox.gpt_neox.layers[l].input_layernorm.weight
        state_dict[f"blocks.{l}.ln1.b"] = neox.gpt_neox.layers[l].input_layernorm.bias

        # For some inexplicable reason, NeoX both uses the concatenated QKV matmul of GPT-2 (afaict this has a neglible performance impact) AND has the flattened axis in the DIFFERENT order of (head_index qkv d_head) - this took me an hour to debug...
        W = neox.gpt_neox.layers[l].attention.query_key_value.weight
        W = einops.rearrange(W, "(i qkv h) m->qkv i m h", i=cfg.n_heads, qkv=3)

        # Fold in layer norm weights
        state_dict[f"blocks.{l}.attn.W_Q"] = W[0]
        state_dict[f"blocks.{l}.attn.W_K"] = W[1]
        state_dict[f"blocks.{l}.attn.W_V"] = W[2]

        qkv_bias = neox.gpt_neox.layers[l].attention.query_key_value.bias
        qkv_bias = einops.rearrange(
            qkv_bias,
            "(index qkv head)->qkv index head",
            qkv=3,
            index=cfg.n_heads,
            head=cfg.d_head,
        )
        # Fold in layer norm biases
        state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0]
        state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1]
        state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2]
        
        W_O = neox.gpt_neox.layers[l].attention.dense.weight
        W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = neox.gpt_neox.layers[l].attention.dense.bias
        
        state_dict[f"blocks.{l}.ln2.w"] = neox.gpt_neox.layers[l].post_attention_layernorm.weight
        state_dict[f"blocks.{l}.ln2.b"] = neox.gpt_neox.layers[l].post_attention_layernorm.bias

        state_dict[f"blocks.{l}.mlp.W_in"] = neox.gpt_neox.layers[l].mlp.dense_h_to_4h.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = neox.gpt_neox.layers[l].mlp.dense_h_to_4h.bias
        
        state_dict[f"blocks.{l}.mlp.W_out"] = neox.gpt_neox.layers[l].mlp.dense_4h_to_h.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = neox.gpt_neox.layers[l].mlp.dense_4h_to_h.bias
    state_dict["ln_final.w"] = neox.gpt_neox.final_layer_norm.weight
    state_dict["ln_final.b"] = neox.gpt_neox.final_layer_norm.bias
    
    state_dict["unembed.W_U"] = neox.embed_out.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab)
    return state_dict

def convert_opt_weights(opt, cfg: EasyTransformerConfig):
    state_dict = {}

    state_dict["embed.W_E"] = opt.model.decoder.embed_tokens.weight
    state_dict["pos_embed.W_pos"] = opt.model.decoder.embed_positions.weight[2:, :]

    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = opt.model.decoder.layers[l].self_attn_layer_norm.weight
        state_dict[f"blocks.{l}.ln1.b"] = opt.model.decoder.layers[l].self_attn_layer_norm.bias
        
        W_Q = opt.model.decoder.layers[l].self_attn.q_proj.weight
        W_K = opt.model.decoder.layers[l].self_attn.k_proj.weight
        W_V = opt.model.decoder.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(
            W_Q,
            "(index d_head) d_model->index d_model d_head",
            index=cfg.n_heads,
        )
        W_K = einops.rearrange(
            W_K,
            "(index d_head) d_model->index d_model d_head",
            index=cfg.n_heads,
        )
        W_V = einops.rearrange(
            W_V,
            "(index d_head) d_model->index d_model d_head",
            index=cfg.n_heads,
        )

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_K"] = W_K
        state_dict[f"blocks.{l}.attn.W_V"] = W_V

        q_bias = einops.rearrange(
            opt.model.decoder.layers[l].self_attn.q_proj.bias,
            "(head_index d_head)->head_index d_head",
            head_index=cfg.n_heads,
            d_head=cfg.d_head,
        )
        k_bias = einops.rearrange(
            opt.model.decoder.layers[l].self_attn.k_proj.bias,
            "(head_index d_head)->head_index d_head",
            head_index=cfg.n_heads,
            d_head=cfg.d_head,
        )
        v_bias = einops.rearrange(
            opt.model.decoder.layers[l].self_attn.v_proj.bias,
            "(head_index d_head)->head_index d_head",
            head_index=cfg.n_heads,
            d_head=cfg.d_head,
        )

        state_dict[f"blocks.{l}.attn.b_Q"] = q_bias
        state_dict[f"blocks.{l}.attn.b_K"] = k_bias
        state_dict[f"blocks.{l}.attn.b_V"] = v_bias

        W_O = opt.model.decoder.layers[l].self_attn.out_proj.weight
        W_O = einops.rearrange(
            W_O,
            "d_model (index d_head)->index d_head d_model",
            index=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = opt.model.decoder.layers[
            l
        ].self_attn.out_proj.bias

        state_dict[f"blocks.{l}.ln2.w"] = opt.model.decoder.layers[l].final_layer_norm.weight
        state_dict[f"blocks.{l}.ln2.b"] = opt.model.decoder.layers[l].final_layer_norm.bias
        
        state_dict[f"blocks.{l}.mlp.W_in"] = opt.model.decoder.layers[l].fc1.weight.T
        state_dict[f"blocks.{l}.mlp.W_out"] = opt.model.decoder.layers[l].fc2.weight.T
        
        state_dict[f"blocks.{l}.mlp.b_in"] = opt.model.decoder.layers[l].fc1.bias
        state_dict[f"blocks.{l}.mlp.b_out"] = opt.model.decoder.layers[l].fc2.bias
    state_dict[f"ln_final.w"] = opt.model.decoder.final_layer_norm.weight
    state_dict[f"ln_final.b"] = opt.model.decoder.final_layer_norm.bias
    state_dict["unembed.W_U"] = opt.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab)
    return state_dict