import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer

MODEL_NAME = "Qwen/Qwen3-0.6B"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", trust_remote_code=True
    )
    hf_model.eval()

    tl_model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME)
    tl_model.eval()

    tokens = tokenizer("Hello world", return_tensors="pt")["input_ids"]

    with torch.no_grad():
        hf_out = hf_model(tokens, output_hidden_states=True)
        tl_logits, cache = tl_model.run_with_cache(tokens)

    for i in range(tl_model.cfg.n_layers):
        diff = (
            cache[f"blocks.{i}.hook_resid_post"] - hf_out.hidden_states[i + 1]
        ).abs().max()
        print(f"Layer {i}: max hidden state diff {diff.item():.6f}")

    logit_diff = (tl_logits - hf_out.logits).abs().max()
    print(f"Max logit diff: {logit_diff.item():.6f}")


if __name__ == "__main__":
    main()
