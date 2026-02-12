"""Example: Generate text with OpenELM via TransformerBridge.

Note: OpenELM-1_1B is a small (1.1B param) base model. Generation quality is
limited compared to larger or instruction-tuned models. Base models work best
when continuing longer passages rather than short prompts. The bridge reproduces
the native HF model logits exactly (diff = 0.0, perplexity ~10.4).

OpenELM's model card recommends repetition_penalty=1.2 for coherent output.
"""

from transformer_lens.model_bridge.bridge import TransformerBridge

model = TransformerBridge.boot_transformers(
    "apple/OpenELM-1_1B",
    trust_remote_code=True,
)

# Base models generate best with longer context
print("=== Document continuation ===")
print(
    model.generate(
        "Paris is the capital and most populous city of France. Since the 17th century, "
        "Paris has been one of the world's major centres of finance, diplomacy, commerce, "
        "fashion, gastronomy, and science. The city is known for",
        max_new_tokens=80,
        temperature=0.7,
        top_k=40,
        repetition_penalty=1.2,
    )
)

print("\n=== Code completion ===")
print(
    model.generate(
        "The following Python function computes the factorial of a number:\n\n"
        "def factorial(n):\n"
        '    """Return the factorial of n."""\n'
        "    if n == 0:\n"
        "        return 1\n"
        "    return n *",
        max_new_tokens=60,
        temperature=0.7,
        top_k=40,
        repetition_penalty=1.2,
    )
)

print("\n=== Story continuation ===")
print(
    model.generate(
        "Chapter 1: The Beginning\n\n"
        "It was a dark and stormy night when the old professor first arrived at "
        "the university. He carried with him a leather satchel full of ancient "
        "manuscripts, each one more mysterious than the last. As he walked through "
        "the empty corridors, he noticed",
        max_new_tokens=80,
        temperature=0.7,
        top_k=40,
        repetition_penalty=1.2,
    )
)

print("\n=== Short prompt (greedy) ===")
print(
    model.generate(
        "The quick brown fox",
        max_new_tokens=30,
        do_sample=False,
        repetition_penalty=1.2,
    )
)
