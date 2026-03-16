# test_hubert_ctc_lmhead.py
"""
Test script to verify HookedAudioEncoder.forward(..., use_ctc=True)
loads/uses an lm_head and produces CTC logits.

Usage:
    python test_hubert_ctc_lmhead.py
Change the import to point at your HookedAudioEncoder implementation.
"""

import math

import numpy as np
import torch

from transformer_lens import HookedAudioEncoder

# ----- CONFIG -----
SAMPLE_RATE = 16000
DURATION_S = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
# If you want to attempt optional decoding with a HF tokenizer,
# set TOKENIZER_NAME to a valid tokenizer (e.g. "facebook/wav2vec2-base-960h")
# or set to None to skip tokenizer decoding.
TOKENIZER_NAME = "facebook/hubert-large-ls960-ft"
# ------------------


def make_sine(frequency=440.0, sr=SAMPLE_RATE, duration=DURATION_S, amplitude=0.1):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return amplitude * np.sin(2 * math.pi * frequency * t)


def has_lm_head(model):
    return any(
        name.endswith("lm_head") or name == "lm_head" for name, _ in model.named_children()
    ) or hasattr(model, "lm_head")


def try_get_lm_head(model):
    if hasattr(model, "lm_head"):
        return model.lm_head
    # try common nested names
    for name, module in model.named_modules():
        if name.endswith("lm_head") or name == "lm_head":
            return module
    return None


def print_param_info(module, prefix=""):
    if module is None:
        print(prefix + "None")
        return
    params = list(module.parameters())
    print(prefix + f"module type: {type(module)}, #params: {sum(p.numel() for p in params)}")
    # print weight shape if available
    if hasattr(module, "weight"):
        try:
            print(prefix + f" weight.shape = {tuple(module.weight.shape)}")
        except Exception:
            pass


if __name__ == "__main__":
    model = HookedAudioEncoder.from_pretrained("facebook/hubert-large-ls960-ft")

    model.to(DEVICE)

    # sample waveform
    wav = make_sine(frequency=440.0)
    x = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)  # shape (1, T)

    print("=== lm_head presence BEFORE forward() ===")
    print("has_lm_head():", has_lm_head(model))
    print("try_get_lm_head():")
    print_param_info(try_get_lm_head(model), prefix="  ")

    # Forward pass with use_ctc=True (some model APIs accept it directly, some do not).
    print(
        "\nCalling forward(..., use_ctc=True) -- if that fails, will set attribute and call without arg"
    )
    logits = None
    forward_exc = None
    try:
        # try direct call with argument
        out = model(x, use_ctc=True)
    except TypeError as e:
        # forward signature may not accept use_ctc param; try setting attribute on model and call
        forward_exc = e
        print(
            "Direct forward(..., use_ctc=True) failed with TypeError - will try setting model.use_ctc = True and calling forward(x)."
        )
        try:
            if hasattr(model, "use_ctc"):
                model.use_ctc = True
            else:
                # set attribute anyway
                setattr(model, "use_ctc", True)
            out = model(x)
        except Exception as e2:
            print("Forward still failed after setting model.use_ctc =", e2)
            raise

    # Normalize out to logits tensor if possible
    def extract_logits(out):
        if out is None:
            return None
        if isinstance(out, torch.Tensor):
            return out  # assume logits
        # dict-like outputs: look for common keys
        if isinstance(out, dict):
            for key in ("logits", "ctc_logits", "predictions", "hidden_states"):
                if key in out:
                    t = out[key]
                    # if hidden_states is (batch, seq, dim) that's also fine to inspect
                    if isinstance(t, torch.Tensor):
                        return t
            # if no known keys found, try to pick first tensor value
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    return v
        # fallback: try to convert
        return None

    logits = extract_logits(out)
    print("\n=== Post-forward lm_head presence ===")
    print("has_lm_head():", has_lm_head(model))
    lm = try_get_lm_head(model)
    print("try_get_lm_head():")
    print_param_info(lm, prefix="  ")

    if logits is None:
        print(
            "\nCould not automatically extract logits from the model output. The model returned:",
            type(out),
        )
        # if out is tensor-like but not torch tensor, attempt conversion
        if hasattr(out, "numpy"):
            try:
                logits = torch.from_numpy(out.numpy()).to(DEVICE)
            except Exception:
                pass

    if logits is not None:
        print("\n=== Logits / CTC output info ===")
        print("logits type:", type(logits))
        print("logits shape:", tuple(logits.shape))
        # typical CTC logits shape: (batch, time, vocab_size) or (batch, seq_len, vocab)
        try:
            print(
                "stats: min=%.6g max=%.6g mean=%.6g"
                % (logits.min().item(), logits.max().item(), logits.mean().item())
            )
        except Exception:
            pass
        assert torch.isfinite(logits).all(), "Found NaNs/Infs in logits!"

        # simple decode: argmax over last dim -> token ids
        if logits.ndim >= 2:
            token_dim = -1
            token_ids = logits.argmax(dim=token_dim)  # shape: (batch, time)
            token_ids_cpu = token_ids.detach().cpu().numpy()
            print("Sample argmax token ids (first batch, up to first 40 frames):")
            print(token_ids_cpu[0][:40].tolist())

            # Optional: try to decode token ids to text if a tokenizer is available
            if TOKENIZER_NAME is not None:
                try:
                    from transformers import AutoTokenizer

                    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
                    # For many CTC tokenizers, you need to collapse repeats and remove blank token id (often id=0 or tok.pad_token_id)
                    # Here we do a naive collapse+remove assuming blank token is tokenizer.pad_token_id or tokenizer.pad_token_id==tok.pad_token_id
                    blank_id = getattr(tok, "pad_token_id", None)
                    seq = token_ids_cpu[0].tolist()
                    # collapse repeats and remove blanks
                    collapsed = []
                    prev = None
                    for t in seq:
                        if t == prev:
                            prev = t
                            continue
                        prev = t
                        if blank_id is not None and t == blank_id:
                            continue
                        collapsed.append(t)
                    decoded = tok.decode(collapsed, skip_special_tokens=True)
                    print("Decoded (naive collapse) text:", decoded)
                except Exception as e:
                    print("Optional decoding failed:", e)

    else:
        print("No logits found — cannot run CTC-specific checks.")

        # Gradient test specifically for transformer encoder (since lm_head is frozen)
    print("\nRunning gradient propagation test through transformer encoder...")

    model.train()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    try:
        out2 = model(x, use_ctc=True)
    except TypeError:
        if hasattr(model, "use_ctc"):
            model.use_ctc = True
        out2 = model(x)

    logits2 = extract_logits(out2)
    if logits2 is None:
        print("Could not extract logits for gradient test; aborting gradient check.")
    else:
        loss = logits2.mean()
        loss.backward()

        # --- Check that lm_head is frozen ---
        lm = try_get_lm_head(model)
        if lm is not None:
            lm_params = list(lm.parameters())
            grads = [p.grad for p in lm_params if p.grad is not None]
            if len(grads) > 0:
                print("Warning: lm_head has gradients, but it should be frozen (eval mode).")
            else:
                print("✅ lm_head correctly frozen (no gradients).")

        # --- Check that transformer block parameters have gradients ---
        has_transformer_grad = False
        for name, p in model.named_parameters():
            if "transformer" in name or "encoder" in name or "block" in name:
                print(name)
                if p.grad is not None and torch.isfinite(p.grad).all():
                    has_transformer_grad = True
                    break

        if has_transformer_grad:
            print("✅ Gradient test PASSED: transformer block parameters have finite gradients.")
        else:
            print("❌ Gradient test FAILED: no gradients found in transformer blocks.")

    print("\n=== DONE ===")
    print("Interpretation notes:")
    print(
        " - If lm_head appears AFTER calling forward(use_ctc=True) and logits shape looks like (B, T, V),"
    )
    print(
        "   then your forward-path is constructing/attaching an lm_head and producing CTC logits."
    )
    print(
        " - If lm_head parameters have finite gradients after loss.backward(), the head is hooked into the graph."
    )
    print(
        " - If you want a numeric golden-check, instantiate a HF Hubert/Wav2Vec2 CTC model and compare pooled logits/ids (optional)."
    )
    print(model.named_parameters())
