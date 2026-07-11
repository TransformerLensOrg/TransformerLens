import torch
from transformers import ASTConfig, ASTForAudioClassification

from transformer_lens.model_bridge import TransformerBridge

def test_ast_parity():
    # 1. setup a tiny HF AST config for instantaneous local testing
    hf_config = ASTConfig(
        hidden_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=24,
        patch_size=16,
    )

    # explicitly define the architecture list for the local config
    hf_config.architectures = ["ASTForAudioClassification"]

    hf_model = ASTForAudioClassification(hf_config)
    hf_model.eval()

    # 2. boot the v3 transformerbridge natively
    # bridge auotmatically detecty ASTForAudioClassification and use new ASTArchitectureAdapater
    tl_model = TransformerBridge.boot_transformers("ast", hf_model=hf_model)
    tl_model.eval()

    # 3. create dummy spectrogram input [batch, freq, time]
    dummy_spectrogram = torch.randn(1, 1024, 128)

    # 4. compare forward passes
    with torch.no_grad():
        hf_logits = hf_model(dummy_spectrogram).logits

        # bridge forward pass (extract sequence before HF's pooling)
        _, cache = tl_model.run_with_cache(dummy_spectrogram)
        resid = cache["ln_final.hook_normalized"]

        # apply AST pooling logic to the bridges residual stream
        cls_token = resid[:, 0, :]
        dist_token = resid[:, 1, :]
        pooled_out = (cls_token + dist_token) / 2.0

        # apply the final classifier (which holds 2nd layernorm + dense)
        tl_logits = hf_model.classifier(pooled_out)
    
    diff = (hf_logits - tl_logits).abs().max().item()
    print(f"Briddge vs HF Max Logit Diff: {diff:.6e}")

    # 5. assert parity
    assert diff < 1e-4, "Parity failed: Bridge tensors do not match HuggingFace."
    print("Parity dub. V3 Bridge connected excellent")

if __name__ == "__main__":
    test_ast_parity()