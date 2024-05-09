# Tutorials

- **Start with the [main demo](https://neelnanda.io/transformer-lens-demo) to learn how the library works, and the basic features**.

## Where To Start

- To see what using it for exploratory analysis in practice looks like, check out [my notebook analysing Indirect Objection Identification](https://neelnanda.io/exploratory-analysis-demo) or [my recording of myself doing research](https://www.youtube.com/watch?v=yo4QvDn-vsU)!

- [What is a Transformer tutorial](https://neelnanda.io/transformer-tutorial)

## Demos

- [**Activation Patching in TransformerLens**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Activation_Patching_in_TL_Demo.ipynb) - Accompanies the [Exploratory Analysis Demo](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Exploratory Analysis Demo.ipynb). This demo explains how to use [Activation Patching](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=qeWBvs-R-taFfcCq-S_hgMqx) in TransformerLens, a mechanistic interpretability technique that uses causal intervention to identify which activations in a model matter for producing an output.

- [**Attribution Patching**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Attribution_Patching_Demo.ipynb) - [Attribution Patching](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) is an incomplete project that uses gradients to take a linear approximation to activation patching. It's a good approximation when patching in small activations like the outputs of individual attention heads, and bad when patching in large activations like a residual stream.

- [**Exploratory Analysis**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Exploratory_Analysis_Demo.ipynb) - Probably the best place to start, after the Main Demo. Demonstrates how to use TransformerLens to perform exploratory analysis - focuses less on rigor and more on getting a grasp of what's going on quickly. Uses a lot of useful interpretability techniques like logit attribution and activation patching. Steal liberally from this!

- [**Grokking**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Grokking_Demo.ipynb) - "Grokking" is a phenomenon where a model can learn to memorise the training data (minimising training loss) but then, if trained for a lot longer, can learn to generalise, leading to a sharp decrease in test loss as well. This demo shows training a model on the task of modular addition, verifying that it groks, and doing analysis. The demo is light on explanation, so you'll probably want to pair it with [Neel's video series](https://www.youtube.com/watch?v=ob4vuiqG2Go) on the paper it's based on.

- [**Head Detector**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Head_Detector_Demo.ipynb) - Shows how to use TransformerLens to automatically detect several common types of attention head, as well as create your own custom detection algorithms to find your own!

- [**Interactive Neuroscope**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb) - Very hacky demo, but this is a feature, not a bug. Shows how to quickly create useful web-based visualisations of data, even if you're not a professional front-end developer. This demo creates an interactive Neuroscope - a visualization of a neuron's activations on text that will dynamically update as you edit the text.

- [**LLaMA**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/LLaMA.ipynb) - Converts Meta's LLaMA model (7B parameter version for now until multi-GPU support is added) to TransformerLens.

- [**Main Demo**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Main_Demo.ipynb) - The main demo. This is where to start if you're new to TransformerLens. Shows a lot of great features for getting started, including available models, how to access model activations, and generally useful features you should know about.

- [**No Position Experiment**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/No_Position_Experiment.ipynb) - The accompanying notebook to Neel's [real-time research video](https://www.youtube.com/watch?v=yo4QvDn-vsU). Trains a model with no positional embeddings to predict the previous token, and makes a start at analysing what's going on there!

- [**Othello-GPT**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Othello_GPT.ipynb) - This is a demo notebook porting the weights of the Othello-GPT Model from the excellent [Emergent World Representations](https://arxiv.org/pdf/2210.13382.pdf) paper to TransformerLens. Neel's [sequence on investigating this](https://www.lesswrong.com/s/nhGNHyJHbrofpPbRG) is also well worth reading if you're interested in this topic!

- [**SVD Interpreter Demo**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/SVD_Interpreter_demo.ipynb) - Based on the [Conjecture post](https://www.lesswrong.com/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight#Directly_editing_SVD_representations) about how the singular value decompositions of transformer matrices are surprisingly interpretable, this demo shows how to use TransformerLens to reproduce this and investigate further.

- [**Tracr to TransformerLens**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Tracr_to_Transformer_Lens_Demo.ipynb) - [Tracr](https://github.com/deepmind/tracr) is a cool new DeepMind tool that compiles a written program in [RASP](https://arxiv.org/abs/2106.06981) to transformer weights.This is a (hacky!) script to convert Tracr weights from the JAX form to a TransformerLens HookedTransformer in PyTorch.
