"""`benchmark_generation` tests the generation loop, not the model's opinions.

Two real behaviours it must survive, both measured on checkpoints:

* EXAONE-4 scores EOS as the argmax continuation of a bare test prompt (raw HF
  does too). That is the model choosing to stop, not a stalled loop.
* StarCoder2's BOS *is* its EOS and its tokenizer never prepends one; handing
  it a leading BOS turns coherent output into "## 1.1.1.1...".

So the benchmark disables EOS stopping and leaves tokenization to the model.
"""

from types import SimpleNamespace

import torch

from transformer_lens.benchmarks.generation import benchmark_generation
from transformer_lens.model_bridge import TransformerBridge

PROMPT = "Natural language processing tasks are typically supervised."
CONTINUATION = " and it continues onward"
DEGENERATE = " 1.1.1.1.1.1"


class _WordTokenizer:
    """Whitespace tokenizer that round-trips: decode(encode(t)) == t.

    Faithfulness matters — the benchmark compares generated text against the
    decoded prompt, so a lossy stub would mask the stall it must detect.
    """

    def __init__(self) -> None:
        self.words: list[str] = []

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self.words:
                self.words.append(word)
            ids.append(self.words.index(word))
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.words[int(i)] for i in ids)


def _bridge(*, stalls_regardless: bool = False) -> TransformerBridge:
    """Stub bridge combining both measured behaviours.

    An uninitialized instance rather than a namespace: the benchmark is
    beartype-checked against TransformerBridge, and booting a real model would
    test the checkpoint rather than the benchmark's contract.
    """
    tokenizer = _WordTokenizer()

    def to_tokens(text):
        return torch.tensor(tokenizer.encode(text)).unsqueeze(0)

    def generate(text, max_new_tokens=10, temperature=1.0, stop_at_eos=True, **kwargs):
        if kwargs.get("prepend_bos"):
            return text + DEGENERATE  # StarCoder2 given a BOS it never wants
        if stalls_regardless or stop_at_eos:
            return text  # EXAONE-4: EOS is the argmax at step 0
        return text + CONTINUATION

    bridge = object.__new__(TransformerBridge)
    bridge.cfg = SimpleNamespace(model_name="fake/exaone-like", architecture="FakeForCausalLM")
    bridge.adapter = SimpleNamespace(supports_generation=True)
    bridge.tokenizer = tokenizer
    bridge.to_tokens = to_tokens
    bridge.generate = generate
    return bridge


def test_eos_at_step_zero_is_not_reported_as_a_stalled_loop() -> None:
    """The EXAONE-4 case: an EOS at step 0 is the model choosing to stop, not a stalled loop."""
    result = benchmark_generation(_bridge(), PROMPT)
    assert result.passed, f"{result.message} / {result.details}"
    assert result.details is not None
    assert result.details["output_tokens"] > result.details["input_tokens"]


def test_model_tokenization_is_not_overridden() -> None:
    """The StarCoder2 case: forcing a BOS overrides adapters that set
    default_prepend_bos=False on purpose, and derails BOS==EOS checkpoints —
    so benchmark_generation must never inject prepend_bos into generate()."""
    bridge = _bridge()
    stub_generate = bridge.generate
    seen: dict = {}

    def recording_generate(text, **kwargs):
        seen.update(kwargs)
        return stub_generate(text, **kwargs)

    bridge.generate = recording_generate
    result = benchmark_generation(bridge, PROMPT)
    assert result.passed, f"{result.message} / {result.details}"
    # Had the benchmark forced a BOS the stub would emit the degenerate
    # trajectory; the decisive check is that no prepend_bos was injected.
    assert not seen.get("prepend_bos")


def test_a_genuinely_stalled_model_still_fails() -> None:
    """The check must not be defanged: a loop that emits nothing even with
    stopping disabled is a real defect and must still be reported."""
    result = benchmark_generation(_bridge(stalls_regardless=True), PROMPT)
    assert not result.passed
    assert "no new tokens" in result.message
