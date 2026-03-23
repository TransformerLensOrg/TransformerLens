import math

import numpy as np
import torch

import transformer_lens.utils as utils
from transformer_lens import HookedAudioEncoder

SAMPLE_RATE = 16000
DURATION_S = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_sine(sr=SAMPLE_RATE, duration=DURATION_S, freq=440.0, amp=0.1):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False, dtype=np.float32)
    return amp * np.sin(2 * math.pi * freq * t)

audio_model = HookedAudioEncoder.from_pretrained("facebook/hubert-base-ls960", device="cuda")

def main():
    wav = make_sine()
    raw_batch = [wav]  # batch of one

    try:
        frames, frame_mask = audio_model.to_frames(raw_batch, sampling_rate=SAMPLE_RATE, move_to_device=True)
    except NameError:
        raise RuntimeError("Replace `audio_model` with your model/wrapper instance that implements to_frames().")

    print("frames.shape:", tuple(frames.shape))
    if frame_mask is not None:
        print("frame_mask.shape:", tuple(frame_mask.shape))

    logits, cache = audio_model.run_with_cache(frames, one_zero_attention_mask=frame_mask, remove_batch_dim=True)
    layer_to_visualize = 0
    pattern_name = utils.get_act_name("pattern", layer_to_visualize)  # e.g. "pattern_0" depending on utils
    try:
        attention_pattern = cache[pattern_name]   # expected shape: (pos, pos, n_heads) or (pos, n_heads, pos) depending on implementation
    except Exception:
        try:
            attention_pattern = cache["pattern", layer_to_visualize, "attn"]
        except Exception as exc:
            raise RuntimeError(f"Couldn't find attention pattern in cache. Keys: {list(cache.keys())}") from exc
    n_frames = attention_pattern.shape[0]
    frame_tokens = [f"f{i}" for i in range(n_frames)]

    print("Layer", layer_to_visualize, "attention pattern shape:", tuple(attention_pattern.shape))
    print("Displaying attention patterns (layer", layer_to_visualize, ")")

    head_index_to_ablate = 0
    layer_to_ablate = 0
    v_act_name = utils.get_act_name("v", layer_to_ablate)

    def head_ablation_hook(value, hook):
        """
        value expected shape: [batch pos head d_head] OR [pos head d_head] when remove_batch_dim=True
        We'll allow both shapes.
        """
        v = value.clone()
        if v.ndim == 4:
            v[:, :, head_index_to_ablate, :] = 0.0
        elif v.ndim == 3:
            v[:, head_index_to_ablate, :] = 0.0
        else:
            raise RuntimeError(f"Unexpected v tensor ndim={v.ndim}")
        return v

    def run_and_get_repr(frames, frame_mask, hooks=None):
        if hooks is None:
            cache = audio_model.run_with_cache(frames, one_zero_attention_mask=frame_mask, remove_batch_dim=True)
            out = audio_model.run_with_hooks(frames, fwd_hooks=[])
        else:
            out = audio_model.run_with_hooks(frames, fwd_hooks=hooks, one_zero_attention_mask=frame_mask)
        logits = None
        if isinstance(out, dict):
            for k in ("logits", "ctc_logits", "logits_ctc", "predictions"):
                if k in out and isinstance(out[k], torch.Tensor):
                    logits = out[k]
                    break
        elif isinstance(out, torch.Tensor):
            logits = out

        if logits is not None:
            if logits.ndim == 3:
                pooled = logits.mean(dim=1)  # (batch, vocab)
            elif logits.ndim == 2:
                pooled = logits  # maybe (batch, vocab)
            else:
                pooled = logits.view(logits.shape[0], -1).mean(dim=1, keepdim=True)
            return pooled, logits, None  # third slot reserved for cache
        try:
            last_layer = audio_model.cfg.n_layers - 1
            resid_name = utils.get_act_name("resid_post", last_layer)
            cache = audio_model.run_with_cache(frames, one_zero_attention_mask=frame_mask, remove_batch_dim=True)
            resid = cache[resid_name]  # e.g. (pos, d) or (batch,pos,d)
            if resid.ndim == 3:
                pooled = resid.mean(dim=1)  # (batch, d)
            elif resid.ndim == 2:
                pooled = resid.mean(dim=0, keepdim=True)
            else:
                raise RuntimeError("Unexpected resid_post shape")
            return pooled, None, cache
        except Exception as e:
            raise RuntimeError("Couldn't extract logits or resid_post; adapt the extraction to your model's output format.") from e

    baseline_repr, baseline_logits, baseline_cache = run_and_get_repr(frames, frame_mask, hooks=None)
    print("Baseline representation shape:", tuple(baseline_repr.shape))

    hooks = [(v_act_name, head_ablation_hook)]
    ablated_repr, ablated_logits, ablated_cache = run_and_get_repr(frames, frame_mask, hooks=hooks)
    print("Ablated representation shape:", tuple(ablated_repr.shape))

    cos = torch.nn.functional.cosine_similarity(baseline_repr, ablated_repr, dim=-1)
    print("Cosine similarity baseline vs ablated:", cos.detach().cpu().numpy())

    if baseline_logits is not None and ablated_logits is not None:
        b_ids = baseline_logits.argmax(dim=-1)  # (batch, time)
        a_ids = ablated_logits.argmax(dim=-1)
        print("Sample argmax token ids (baseline):", b_ids[0][:40].cpu().numpy().tolist())
        print("Sample argmax token ids (ablated): ", a_ids[0][:40].cpu().numpy().tolist())

    print("Done. Interpret the results:")
    print(" - A large drop in cosine similarity (or large change in argmax tokens / increase in loss) means the ablated head mattered.")
    print(" - If ablation causes little change, that head may be redundant or not used for this example.")

if __name__ == "__main__":
    main()
