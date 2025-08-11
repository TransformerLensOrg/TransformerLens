from transformer_lens import HookedTransformer

def test_generate_batch():
    """
    Test that batched and individual prompt generation produce the same outputs.
    """
    model = HookedTransformer.from_pretrained("gpt2")
    input_prompts = ["Hello, my dog is cute", "This is a much longer text. Hello, my cat is cute"]
    orig_outputs = []
    for prompt in input_prompts:
        out = model.generate(prompt, verbose=False, do_sample=False)
        orig_outputs.append(out)
        
    batched_outputs = model.generate(input_prompts, verbose=False, do_sample=False)
    for i in range(len(orig_outputs)):
        assert orig_outputs[i] == batched_outputs[i]