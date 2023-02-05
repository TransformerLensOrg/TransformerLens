from typeguard.importhook import install_import_hook

install_import_hook("transformer_lens")

from transformer_lens import HookedTransformer
from torchtyping import TensorType as TT, patch_typeguard

patch_typeguard()

MODEL = "solu-1l"

prompt = "Hello World!"
model = HookedTransformer.from_pretrained(MODEL)

act_names_in_cache = ['hook_embed', 'hook_pos_embed', 'blocks.0.hook_resid_pre', 'blocks.0.ln1.hook_scale', 'blocks.0.ln1.hook_normalized', 'blocks.0.attn.hook_q', 'blocks.0.attn.hook_k', 'blocks.0.attn.hook_v', 'blocks.0.attn.hook_attn_scores', 'blocks.0.attn.hook_pattern', 'blocks.0.attn.hook_z', 'blocks.0.hook_attn_out', 'blocks.0.hook_resid_mid', 'blocks.0.ln2.hook_scale', 'blocks.0.ln2.hook_normalized', 'blocks.0.mlp.hook_pre', 'blocks.0.mlp.hook_mid', 'blocks.0.mlp.ln.hook_scale', 'blocks.0.mlp.ln.hook_normalized', 'blocks.0.mlp.hook_post', 'blocks.0.hook_mlp_out', 'blocks.0.hook_resid_post', 'ln_final.hook_scale', 'ln_final.hook_normalized']

def test_cache_hook_names():
    logits, cache = model.run_with_cache(prompt)
    assert list(cache.keys()) == act_names_in_cache

