# Further Details on Config Options
## Shortformer Attention (`positional_embeddings_type == "shortformer"`)
Shortformer style models are a variant on GPT-2 style positional embeddings, which do not add positional embeddings into the residual stream but instead add it in to the queries and keys immediately before multiplying by W_Q and W_K, and NOT having it around for the values or MLPs. It's otherwise the same - the positional embeddings are absolute, and are learned. The positional embeddings are NOT added to the residual stream in the standard way, and instead the queries and keys are calculated as W_Q(res_stream + pos_embed) and W_K(res_stream + pos_embed). The values and MLPs are calculated as W_V(res_stream) and W_MLP(res_stream) and so don't have access to positional information. This is otherwise the same as GPT-2 style positional embeddings. This is a variant on the Shortformer model from the paper [Shortformer: The Benefits of Shorter Sequences in Language Modeling](https://arxiv.org/abs/2012.15832). It's morally similar to rotary, which also only gives keys & queries access to positional info

The original intention was to use this to do more efficient caching: caching is hard with absolute positional embeddings, since you can't translate the context window without recomputing the entire thing, but easier if the prior values and residual stream terms are the same. I've mostly implemented it because it makes it easier for models to form induction heads. I'm not entirely sure why, though hypothesise that it's because there's two ways for induction heads to form with positional embeddings in the residual stream and only one with shortformer style positional embeddings.
    
## What is LayerNorm Folding? (`fold_ln`)
[LayerNorm](https://wandb.ai/wandb_fc/LayerNorm/reports/Layer-Normalization-in-Pytorch-With-Examples---VmlldzoxMjk5MTk1) is a common regularisation technique used in transformers. Annoyingly, unlike eg BatchNorm, it can't be turned off at inference time, it's a meaningful change to the mathematical function implemented by the transformer. From an interpretability perspective, this is a headache! And it's easy to shoot yourself in the foot by naively ignoring it - eg, making the mistake of saying neuron_pre = resid_mid @ W_in, rather than LayerNorm(resid_mid) @ W_in. This mistake is an OK approximation, but by folding in the LayerNorm we can do much better!

TLDR: If we have LayerNorm (weights w_ln and b_ln) followed by a linear layer (W+b), we can reduce the LayerNorm to LayerNormPre (just centering & normalising) and follow it by a linear layer with `W_eff = w[:, None] * W` (element-wise multiplication) and `b_eff = b + b_ln @ W`. This is computationally equivalent, and it never makes sense to think of W and w_ln as separate objects, so EasyTransformer handles it for you when loading pre-trained weights - set fold_ln = False when loading a state dict if you want to turn this off
        
Mathematically, LayerNorm is the following:
```
x1 = x0 - x0.mean()
x2 = x1 / ((x1**2).mean()).sqrt()
x3 = x2 * w
x4 = x3 + b
```
        
Apart from dividing by the norm, these are all pretty straightforwards operations from a linear algebra perspective. And from an interpretability perspective, if anything is linear, it's really easy and you can mostly ignore it (everything breaks up into sums, you can freely change basis, don't need to track interference between terms, etc) - the hard part is engaging with non-linearities!
        
A key thing to bear in mind is that EVERY time we read from the residual stream, we apply a LayerNorm - this gives us a lot of leverage to reason about it!
        
So let's translate this into linear algebra notation.
        `x0` is a vector in `R^n`

```
x1 = x0 - x0.mean()
   = x0 - (x0.mean()) * ones (broadcasting, ones=torch.ones(n))
   = (x0 @ ones/sqrt(n)) * ones/sqrt(n).
```

ones has norm sqrt(n), so ones/sqrt(n) is the unit vector in the diagonal direction. We're just projecting x0 onto this (fixed) vector and subtracting that value off. Alternately, we're projecting onto the n-1 dimensional subspace orthogonal to ones.
            
Since LayerNorm is applied EVERY time we read from the stream, the model just never uses the ones direction of the residual stream, so it's essentially just decreasing d_model by one. We can simulate this by just centering all matrices writing to the residual stream.

Why is removing this dimension useful? I have no idea! I'm not convinced it is...
        
```
x2 = x1 / ((x1**2).mean()).sqrt() (Ignoring eps)
   = (x1 / x1.norm()) * sqrt(n)
```

This is a projection onto the unit sphere (well, sphere of radius sqrt(n) - the norm of ones). This is fundamentally non-linear, eg doubling the input keeps the output exactly the same.

This is by far the most irritating part of LayerNorm. I THINK it's mostly useful for numerical stability reasons and not used to do useful computation by the model, but I could easily be wrong! And interpreting a circuit containing LayerNorm sounds like a nightmare...

In practice, you can mostly get aware with ignore this and treating the scaling factor as a constant, since it does apply across the entire residual stream for each token - this makes it a "global" property of the model's calculation, so for any specific question it hopefully doesn't matter that much. But when you're considering a sufficiently important circuit that it's a good fraction of the norm of the residual stream, it's probably worth thinking about.

```
x3 = x2 * w
   = x2 @ W_ln
```

(`W_ln` is a diagonal matrix with the weights of the LayerNorm - this is equivalent to element-wise multiplication)
This is really easy to deal with - we're about to be input to a linear layer, and can say `(x2 @ W_ln) @ W = x2 @ (W_ln @ W) = x2 @ W_eff` - we can just fold the LayerNorm weights into the linear layer weights.
        
`x4 = x3 + b` is similarly easy - `x4 @ W + B = x2 @ W_eff + B_eff`, where `W_eff = W_ln @ W` and `B_eff = B + b @ W`
        
This function is calculating `W_eff` and `B_eff` for each layer reading from the residual stream and replacing W and B with those
        
        
See this for more: https://transformer-circuits.pub/2021/framework/index.html#:~:text=Handling%20Layer%20Normalization