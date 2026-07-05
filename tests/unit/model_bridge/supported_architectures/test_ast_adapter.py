import torch
from transformers import ASTForAudioClassification, ASTConfig
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.model_bridge.supported_architectures.ast import ASTAdapter, ASTEmbed

def test_ast_parity():
    print("Loading HuggingFace AST . . .")
    # use tiny config to make the test run instantly locally
    hf_config = ASTConfig(
        hidden_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=24,
        patch_size=16,
    )
    hf_model = ASTForAudioClassification(hf_config)

    print("Booting TransformerLens Bridge . . .")
    tl_config_dict = ASTAdapter.get_config_map(hf_config)
    # safety catch: remove 'is_multimodal' in case
    tl_config_dict.pop("is_multimodal", None)

    # dynamically grab the true sequence length HF generated
    actual_n_ctx = hf_model.state_dict()["audio_spectrogram_transformer.embeddings.position_embeddings"].shape[1]
    tl_config_dict["n_ctx"] = actual_n_ctx

    tl_config = HookedTransformerConfig(**tl_config_dict)

    # 2. boot a blank HookedTransformer
    tl_model = HookedTransformer(tl_config)
    # 3. hot-swap: replace the standard text embedder with our AST patch extractor
    tl_model.embed = ASTEmbed(tl_config)

    # 4. map and load the weights
    mapped_weights = ASTAdapter.convert_weights(hf_model.state_dict(), tl_config)
    tl_model.load_state_dict(mapped_weights, strict=False)

    # lock both models to disable dropout for increased reproducibility
    hf_model.eval()
    tl_model.eval()

    # create dummy mel spectrogram
    dummy_spectrogram = torch.randn(1, 1024, 128)

    print("\n--- INITIATING LAYER BY LAYER AUTOPSY ---")
    with torch.no_grad():

        # 0. HF forward pass with hidden states exposed
        hf_outputs = hf_model.audio_spectrogram_transformer(dummy_spectrogram, output_hidden_states=True)
        hf_hidden_states = hf_outputs.hidden_states
        hf_logits = hf_model(dummy_spectrogram).logits

        # == TL manual matrix forward pass ==
        # STEP 1. embeddings parity
        resid = tl_model.embed(dummy_spectrogram)
        seq_len = resid.shape[1]
        resid = resid + tl_model.pos_embed.W_pos[:seq_len, :]

        diff_embed = (hf_hidden_states[0] - resid).abs().max().item()
        print(f"1. Embeddings Max Diff: {diff_embed:.6e}")

        # STEP 2. transformer blocks parity 
        for i, block in enumerate(tl_model.blocks):
            resid = block(resid)
            # compare against the corresponding HF hidden state (i+1 because index 0 is embeddings)
            diff_block = (hf_hidden_states[i+1] - resid).abs().max().item()
            print(f"2. Block {i} Max Diff:  {diff_block:.6e}")
        
        # STEP 3: final LayerNorm
        resid = tl_model.ln_final(resid)

        # STEP 4: pooling and classifier
        cls_token_out = resid[:, 0, :]
        dist_token_out = resid[:, 1, :]
        pooled_out = (cls_token_out + dist_token_out) / 2.0

        # AST: second LayerNorm applied ONLY to pooled token
        classifier_ln_w = hf_model.state_dict()["classifier.layernorm.weight"]
        classifier_ln_b = hf_model.state_dict()["classifier.layernorm.bias"]

        pooled_out = torch.nn.functional.layer_norm(
            pooled_out,
            normalized_shape=(tl_model.cfg.d_model,),
            weight=classifier_ln_w,
            bias=classifier_ln_b,
            eps=tl_model.cfg.eps
        )

        tl_logits = tl_model.unembed(pooled_out)
    
    diff_final = (hf_logits - tl_logits).abs().max().item()
    print(f"3. Final Logits Max Diff: {diff_final:.6e}")
    print("-----------------------------------------\n")

    assert diff_final < 1e-4, "parity failed: tensors do not match"
    print("Parity dub. adapter mapping math yes")

if __name__ == "__main__":
    test_ast_parity()