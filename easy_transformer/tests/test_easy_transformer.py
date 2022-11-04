# %%
from easy_transformer import EasyTransformer
import pytest

model_names = [
    "attn-only-demo",
    "gpt2-small",
    "opt-125m",
    "gpt-neo-125M",
    "stanford-gpt2-small-a",
    "solu-4l-old",
    "solu-6l",
    "attn-only-3l",
    "pythia",
    "gelu-2l",
]
text = "Hello world!"
""" 
# Code to regenerate loss store
store = {}
for name in model_names:
    model = EasyTransformer.from_pretrained(name, device='cuda')
    loss = model(text,return_type="loss")
    store[name] = loss.item()
print(store)
"""
loss_store = {
    "attn-only-demo": 5.701841354370117,
    "gpt2-small": 5.331855773925781,
    "opt-125m": 4.621748924255371,
    "gpt-neo-125M": 4.900552272796631,
    "stanford-gpt2-small-a": 5.652035713195801,
    "solu-4l-old": 5.6021833419799805,
    "solu-6l": 5.7042999267578125,
    "attn-only-3l": 5.747507095336914,
    "pythia": 4.659344673156738,
    "gelu-2l": 6.501802444458008,
}
# %%
@pytest.mark.parametrize("name,expected_loss", list(loss_store.items()))
def test_model(name, expected_loss):
    # Runs the model on short text and checks if the loss is as expected
    model = EasyTransformer.from_pretrained(name)
    loss = model(text, return_type="loss")
    assert (loss.item() - expected_loss) < 4e-5
