# test_hubert_hooked.py
import math

import numpy as np
import torch

from transformer_lens import HookedAudioEncoder

# ---------- CONFIG ----------
SAMPLE_RATE = 16000
DURATION_S = 1.0
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Name of HF checkpoint to use if you want to compare outputs (optional)
HF_CHECKPOINT = "facebook/hubert-base-ls960"  # optional
# ----------------------------

def make_sine(frequency=440.0, sr=SAMPLE_RATE, duration=DURATION_S, amplitude=0.1):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False, dtype=np.float32)
    wav = amplitude * np.sin(2 * math.pi * frequency * t)
    return wav

def run_basic_sanity_tests(model, waveform_np):
    """Run quick checks: forward pass, shape, finite, deterministic, grad flow."""
    model.to(DEVICE)

    # Prepare tensor: shape (batch, time)
    x = torch.from_numpy(waveform_np).unsqueeze(0).to(DEVICE)  # (1, T)

    # 1) Eval forward: no grad
    model.eval()
    with torch.no_grad():
        out1 = model(x)  # adapt if your API uses return_type="predictions" or similar
    print("Forward (eval) output type:", type(out1))
    try:
        out_tensor = out1 if isinstance(out1, torch.Tensor) else out1["predictions"]
    except Exception:
        out_tensor = out1  # fallback

    print("Output shape:", tuple(out_tensor.shape))
    print("Output stats: min=%.6g max=%.6g mean=%.6g" % (out_tensor.min().item(), out_tensor.max().item(), out_tensor.mean().item()))
    assert torch.isfinite(out_tensor).all(), "Found NaNs or Infs in forward output!"

    # 2) Determinism in eval
    with torch.no_grad():
        out2 = model(x)
    # if model returns dict-like, extract tensor again
    out2_tensor = out2 if isinstance(out2, torch.Tensor) else out2["predictions"]
    if not torch.allclose(out_tensor, out2_tensor, atol=1e-6):
        print("Warning: outputs differ between two eval runs (non-deterministic?), max diff:", (out_tensor - out2_tensor).abs().max().item())
    else:
        print("Determinism test passed (eval mode).")

    # 3) Gradient flow test in train mode
    model.train()
    # zero grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
    out_train = model(x)
    out_train_tensor = out_train if isinstance(out_train, torch.Tensor) else out_train["predictions"]

    # small scalar loss
    loss = out_train_tensor.mean()
    loss.backward()
    # check some parameters got gradients
    grads_found = any((p.grad is not None and torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad)
    assert grads_found, "No finite gradients found on any parameter after backward()"
    print("Gradient check passed: some parameters have finite gradients.")

def optional_compare_to_hf(your_model, waveform_np, sr=SAMPLE_RATE):
    """
    OPTIONAL: compare your_model outputs to Hugging Face's HubertModel outputs.
    This requires transformers to be installed and internet access to download the checkpoint.
    Important: to get a meaningful comparison you must match *exact preprocessing* (resampling,
    normalization, padding/truncation) that the HF model expects and that your model used.
    """
    try:
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
    except Exception as e:
        print("Transformers or feature extractor not available:", e)
        return

    print("Loading Hugging Face HubertModel for optional comparison (may take a while)...")
    hf_feat = Wav2Vec2FeatureExtractor(sampling_rate=sr, do_normalize=True)
    hf_model = HubertModel.from_pretrained(HF_CHECKPOINT).to(DEVICE).eval()

    # Prepare input for HF model
    input_values = hf_feat(waveform_np, sampling_rate=sr, return_tensors="pt").get("input_values")  # (1, T)
    input_values = input_values.to(DEVICE)

    with torch.no_grad():
        hf_outputs = hf_model(input_values).last_hidden_state  # (1, L, D)
    # Pool HF tokens to a single vector (simple mean pooling)
    hf_embedding = hf_outputs.mean(dim=1)  # (1, D)

    # Get your model's representation and pool similarly
    your_model.eval()
    with torch.no_grad():
        your_out = your_model(torch.from_numpy(waveform_np).unsqueeze(0).to(DEVICE))
    your_tensor = your_out if isinstance(your_out, torch.Tensor) else your_out["predictions"]  # shape depends on your model
    # If your output has time dimension, mean-pool across time
    if your_tensor.ndim == 3:
        your_emb = your_tensor.mean(dim=1)
    else:
        your_emb = your_tensor  # assume (1, D) or similar

    # Resize / project if dims differ (simple check)
    if hf_embedding.shape[1] != your_emb.shape[1]:
        print(f"Dimension mismatch (HF {hf_embedding.shape[1]} vs your {your_emb.shape[1]}). "
              "You can compare after projecting to a common dim (not shown).")
        return

    # Cosine similarity
    cos = torch.nn.functional.cosine_similarity(hf_embedding, your_emb, dim=1)
    print("Cosine similarity between HF pooled embedding and your model:", cos.cpu().numpy())

if __name__ == "__main__":
    # Create sample waveform
    wav = make_sine(frequency=440.0, sr=SAMPLE_RATE, duration=DURATION_S)

    # -----------------------
    # Instantiate your model
    # -----------------------
    # Example 1: from_pretrained API (if you implemented it)
    model = HookedAudioEncoder.from_pretrained("facebook/hubert-base-ls960").to(DEVICE)
    # Run tests
    run_basic_sanity_tests(model, wav)
    
    # Optionally compare to HF (network required)
    optional_compare_to_hf(model, wav, sr=SAMPLE_RATE)
