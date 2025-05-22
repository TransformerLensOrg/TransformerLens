import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

MODEL_NAME = "Qwen/Qwen3-4B"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype="float32", trust_remote_code=True
).to("cuda")
hf_model.eval()

tl_model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME).to("cuda").to(torch.float32)
tl_model.eval()

tokens = tokenizer("Hello world", return_tensors="pt")["input_ids"].to("cuda")

with torch.no_grad():
    hf_out = hf_model(tokens, output_hidden_states=True)
    tl_logits, cache = tl_model.run_with_cache(
        tokens, names_filter=lambda name: "resid_pre" in name or "resid_post" in name
    )

for i in range(tl_model.cfg.n_layers + 1):
    if i < tl_model.cfg.n_layers:
        diff = (cache[f"blocks.{i}.hook_resid_pre"] - hf_out.hidden_states[i]).abs().max()
    else:
        diff = (
            (tl_model.ln_final(cache[f"blocks.{i-1}.hook_resid_post"]) - hf_out.hidden_states[-1])
            .abs()
            .max()
        )
    print(f"Layer {i}: max hidden state diff {diff.item():.6f}")


logit_diff = (tl_logits - hf_out.logits).abs().max()
print(f"Max logit diff: {logit_diff.item():.6f}")
