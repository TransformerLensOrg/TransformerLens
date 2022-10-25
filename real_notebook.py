#%% [markdown]
## Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small
# <h1><b>Intro</b></h1>
# This notebook is an implementation of the IOI experiments (some with adjustments from the <a href="https://docs.google.com/presentation/d/19H__CYCBL5F3M-UaBB-685J-AuJZNsXqIXZR-O4j9J8/edit#slide=id.g14659e4d87a_0_290">presentation</a>.
# It should be able to be run as by just git cloning this repo (+ some easy installs).
#
# ### Task
# We're interested in GPT-2 ability to complete sentences like "Alice and Bob went to the store, Alice gave a bottle of milk to"...
# GPT-2 knows that it have to output a name that is not the subject (Alice) and that was present in the context: Bob.
# The first apparition of Alice is called "S" (or sometimes "S1") for "Subject", and Bob is the indirect object ("IO"). Even if the sentences we generate contains the last word of the sentence "Bob", we'll never look at the transformer output here. What's matter is the next-token prediction on the token "to", sometime called the "end" token.
#
# ### Tools
# In this notebook, we define a class `IOIDataset` to handle the generation and utils for this particular dataset.
#
# Refer to the demo of the [`easy_transformer` library](https://github.com/neelnanda-io/Easy-Transformer) here: <a href="https://colab.research.google.com/drive/1MLwJ7P94cizVs2LD8Qwi-vLGSoH-cHxq?usp=sharing">demo with ablation & patching</a>.
#
# Reminder of the circuit:
# <img src="https://i.imgur.com/PPtTQRh.png">