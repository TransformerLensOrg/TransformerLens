# %%
""" 
A file with some rough evals for models - I expect you to be likely better off using the HuggingFace evaluate library if you want to do anything properly, but this is here if you want it and want to eg cheaply and roughly compare models you've trained to baselines.
"""

import torch
import tqdm.auto as tqdm
from datasets import load_dataset
from easy_transformer import EasyTransformer, EasyTransformerConfig, utils
from torch.utils.data import DataLoader

# %%
def make_wiki_data_loader(model, batch_size=8):
    """ 
    Evaluate on Wikitext 2, a dump of Wikipedia articles. (Using the train set because it's larger, I don't really expect anyone to bother with quarantining the validation set nowadays.)

    Note there's likely to be dataset leakage into training data (though I believe GPT-2 was explicitly trained on non-Wikipedia data)
    """
    wiki_data = load_dataset("wikitext", "wikitext-2-v1", split="train")
    print(len(wiki_data))
    dataset = utils.tokenize_and_concatenate(wiki_data, model.tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def make_owt_data_loader(model, batch_size=8):
    """ 
    Evaluate on OpenWebText an open source replication of the GPT-2 training corpus (Reddit links with >3 karma)

    I think the Mistral models were trained on this dataset, so they get very good performance.
    """
    owt_data = load_dataset("stas/openwebtext-10k", split="train")
    print(len(owt_data))
    dataset = utils.tokenize_and_concatenate(owt_data, model.tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def make_code_data_loader(model, batch_size=8):
    """ 
    Evaluate on the CodeParrot dataset, a dump of Python code. All models seem to get significantly lower loss here (even non-code trained models like GPT-2), presumably code is much easier to predict than natural language?
    """
    code_data = load_dataset("codeparrot/codeparrot-valid-v2-near-dedup", split="train")
    print(len(code_data))
    dataset = utils.tokenize_and_concatenate(code_data, model.tokenizer, column_name="content")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

DATASET_NAMES = ['wiki', 'owt', 'code']
DATASET_LOADERS = [make_wiki_data_loader, make_owt_data_loader, make_code_data_loader]

# %%
@torch.inference_mode()
def evaluate_on_dataset(model, data_loader, truncate=100):
    running_loss = 0
    total = 0
    for batch in tqdm.tqdm(data_loader):
        loss = model(batch['tokens'].cuda(), return_type="loss")
        running_loss += loss.item()
        total+=1
        if total > truncate: break
    return running_loss/total

# %%
@torch.inference_mode()
def evaluate(model, truncate=100, batch_size=8):
    losses = {}
    for data_name, data_loader_fn in zip(DATASET_NAMES, DATASET_LOADERS):
        data_loader = data_loader_fn(model, batch_size=batch_size)
        loss = evaluate_on_dataset(model, data_loader, truncate=truncate)
        print(f"{data_name}: {loss}")
        losses[f"{data_name}_loss"]=loss
    return losses

# %%