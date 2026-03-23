import math

import numpy as np
import torch

import transformer_lens.utils as utils
from transformer_lens import HookedAudioEncoder

# ---- Simple sine audio generator ----
SAMPLE_RATE = 16000
DURATION_S = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_sine(sr=SAMPLE_RATE, duration=DURATION_S, freq=440.0, amp=0.1):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False, dtype=np.float32)
    return amp * np.sin(2 * math.pi * freq * t)

audio_model = HookedAudioEncoder.from_pretrained("facebook/hubert-base-ls960", device="cuda")

def main():
    # --- Build a 1s test waveform ---
    wav = make_sine()
    # If to_frames expects numpy or torch, both are accepted by your implementation
    raw_batch = [wav]  # batch of one

    # --- Convert to frames using your helper (you provided to_frames) ---
    # IMPORTANT: use the same sampling_rate you used during training/FT (16k typical)
    try:
        frames, frame_mask = audio_model.to_frames(raw_batch, sampling_rate=SAMPLE_RATE, move_to_device=True)
    except NameError:
        raise RuntimeError("Replace `audio_model` with your model/wrapper instance that implements to_frames().")

    # frames shape expected: (batch, frames, hidden)  ; frame_mask: (batch, frames)  (1/0)
    print("frames.shape:", tuple(frames.shape))
    if frame_mask is not None:
        print("frame_mask.shape:", tuple(frame_mask.shape))

    # --- Run with cache to inspect attention pattern ---
    # remove_batch_dim=True makes cached activations shaped like (pos, ...) for easier visualization (like LLaMA example)
    logits, cache = audio_model.run_with_cache(frames, one_zero_attention_mask=frame_mask, remove_batch_dim=True)

    # Picking a layer and head for visualization
    layer_to_visualize = 0
    # act name for attention pattern — this is the same helper you used earlier
    pattern_name = utils.get_act_name("pattern", layer_to_visualize)  # e.g. "pattern_0" depending on utils
    # some implementations store pattern as (layer, "attn") tuple; utils.get_act_name helps avoid mistakes

    # Extract attention pattern. Adapt this extraction if your cache key structure differs:
    try:
        attention_pattern = cache[pattern_name]   # expected shape: (pos, pos, n_heads) or (pos, n_heads, pos) depending on implementation
    except Exception:
        # fallback: try tuple-key style
        try:
            attention_pattern = cache["pattern", layer_to_visualize, "attn"]
        except Exception as exc:
            raise RuntimeError(f"Couldn't find attention pattern in cache. Keys: {list(cache.keys())}") from exc

    # Build human-friendly "tokens" for frames (e.g. frame indices as strings)
    n_frames = attention_pattern.shape[0]
    frame_tokens = [f"f{i}" for i in range(n_frames)]

    print("Layer", layer_to_visualize, "attention pattern shape:", tuple(attention_pattern.shape))
    print("Displaying attention patterns (layer", layer_to_visualize, ")")
    # display(cv.attention.attention_patterns(tokens=frame_tokens, attention=attention_pattern))

    # --- Define a head ablation hook (zero out a given head's v output) ---
    head_index_to_ablate = 0
    layer_to_ablate = 0

    # Hook target: v (value output) or "pattern" depending on what you'd like to ablate.
    # Using the 'v' activation is a common choice, same form as your LLaMA example.
    v_act_name = utils.get_act_name("v", layer_to_ablate)

    def head_ablation_hook(value, hook):
        """
        value expected shape: [batch pos head d_head] OR [pos head d_head] when remove_batch_dim=True
        We'll allow both shapes.
        """
        # convert to mutable clone (some frameworks give non-writable tensors)
        v = value.clone()
        if v.ndim == 4:
            # (B, pos, heads, d)
            v[:, :, head_index_to_ablate, :] = 0.0
        elif v.ndim == 3:
            # (pos, heads, d)
            v[:, head_index_to_ablate, :] = 0.0
        else:
            raise RuntimeError(f"Unexpected v tensor ndim={v.ndim}")
        return v

    # --- Compute a downstream quantity without ablation ---
    # Choose a metric you care about. Good choices:
    #  - CTC logits (if using use_ctc=True) -> argmax tokens or loss
    #  - Pooled encoder representation (mean of final resid_post) -> cosine similarity
    # We'll implement both: try to extract CTC logits from model output; if not found, use pooled resid_post.

    def run_and_get_repr(frames, frame_mask, hooks=None):
        # hooks: list of (act_name, hook_fn) tuples for run_with_hooks
        if hooks is None:
            # run_with_cache to gather activations
            cache = audio_model.run_with_cache(frames, one_zero_attention_mask=frame_mask, remove_batch_dim=True)
            out = audio_model.run_with_hooks(frames, fwd_hooks=[])
            # NOTE: if your API returns outputs directly from run_with_cache, adapt as needed.
        else:
            # run with hooks and also capture cache
            # run_with_hooks typically returns output (or logits) and optionally a cache depending on your implementation
            out = audio_model.run_with_hooks(frames, fwd_hooks=hooks, one_zero_attention_mask=frame_mask)
            # If return_type="both" isn't supported, you can run run_with_cache and run_with_hooks separately.
        # Try to extract CTC logits from `out` first
        logits = None
        if isinstance(out, dict):
            for k in ("logits", "ctc_logits", "logits_ctc", "predictions"):
                if k in out and isinstance(out[k], torch.Tensor):
                    logits = out[k]
                    break
        elif isinstance(out, torch.Tensor):
            # ambiguous: could be embeddings or logits
            logits = out

        # if logits exist -> pooled logits (mean over time) as representation
        if logits is not None:
            # ensure shape (batch, time, vocab) -> pool over time axis (1)
            if logits.ndim == 3:
                pooled = logits.mean(dim=1)  # (batch, vocab)
            elif logits.ndim == 2:
                pooled = logits  # maybe (batch, vocab)
            else:
                pooled = logits.view(logits.shape[0], -1).mean(dim=1, keepdim=True)
            return pooled, logits, None  # third slot reserved for cache

        # fallback: use final residual activation from cache (resid_post of last layer)
        try:
            last_layer = audio_model.cfg.n_layers - 1
            resid_name = utils.get_act_name("resid_post", last_layer)
            # get cache from run_with_cache (we ran above)
            cache = audio_model.run_with_cache(frames, one_zero_attention_mask=frame_mask, remove_batch_dim=True)
            resid = cache[resid_name]  # e.g. (pos, d) or (batch,pos,d)
            # mean-pool across pos dimension
            if resid.ndim == 3:
                pooled = resid.mean(dim=1)  # (batch, d)
            elif resid.ndim == 2:
                pooled = resid.mean(dim=0, keepdim=True)
            else:
                raise RuntimeError("Unexpected resid_post shape")
            return pooled, None, cache
        except Exception as e:
            raise RuntimeError("Couldn't extract logits or resid_post; adapt the extraction to your model's output format.") from e

    # Get baseline representation
    baseline_repr, baseline_logits, baseline_cache = run_and_get_repr(frames, frame_mask, hooks=None)
    print("Baseline representation shape:", tuple(baseline_repr.shape))

    # --- Run with ablation hook and get representation ---
    hooks = [(v_act_name, head_ablation_hook)]
    ablated_repr, ablated_logits, ablated_cache = run_and_get_repr(frames, frame_mask, hooks=hooks)
    print("Ablated representation shape:", tuple(ablated_repr.shape))

    # --- Compare representations (cosine similarity) ---
    cos = torch.nn.functional.cosine_similarity(baseline_repr, ablated_repr, dim=-1)
    print("Cosine similarity baseline vs ablated:", cos.detach().cpu().numpy())

    # If you have logits, you can also compare token sequences (argmax) or loss increase
    if baseline_logits is not None and ablated_logits is not None:
        b_ids = baseline_logits.argmax(dim=-1)  # (batch, time)
        a_ids = ablated_logits.argmax(dim=-1)
        print("Sample argmax token ids (baseline):", b_ids[0][:40].cpu().numpy().tolist())
        print("Sample argmax token ids (ablated): ", a_ids[0][:40].cpu().numpy().tolist())

    print("Done. Interpret the results:")
    print(" - A large drop in cosine similarity (or large change in argmax tokens / increase in loss) means the ablated head mattered.")
    print(" - If ablation causes little change, that head may be redundant or not used for this example.")

if __name__ == "__main__":
    # create/instantiate your model here: replace the placeholder below
    # Example:
    # audio_model = HookedAudioEncoder.from_pretrained("...").to(DEVICE)
    # audio_model.cfg.device = DEVICE
    # For wrapper that exposes to_frames:
    # audio_model = YourWrapperClass(...)
    main()
