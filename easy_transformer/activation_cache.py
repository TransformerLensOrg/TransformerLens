
import easy_transformer.utils as utils
from easy_transformer.utils import Slice, SliceInput
import torch
import einops
from fancy_einsum import einsum
from typing import Optional, Union
import re
import numpy as np
import logging

class ActivationCache:
    """ 
    A wrapper around a dictionary of cached activations from a model run, with a variety of helper functions. In general, any utility which is specifically about editing/processing activations should be a method here, while any utility which is more general should be a function in utils.py, and any utility which is specifically about model weights should be in EasyTransformer.py or components.py

    WARNING: The biggest footgun and source of bugs in this code will be keeping track of indexes, dimensions, and the numbers of each. There are several kinds of activations:

    Internal attn head vectors: q, k, v, z. Shape [batch, pos, head_index, d_head]
    Internal attn pattern style results: attn (post softmax), attn_scores (pre-softmax). Shape [batch, head_index, query_pos, key_pos]
    Attn head results: result. Shape [batch, pos, head_index, d_model]
    Internal MLP vectors: pre, post, mid (only used for solu_ln - the part between activation + layernorm). Shape [batch, pos, d_mlp]
    Residual stream vectors: resid_pre, resid_mid, resid_post, attn_out, mlp_out, embed, pos_embed, normalized (output of each LN or LNPre). Shape [batch, pos, d_model]
    LayerNorm Scale: scale. Shape [batch, pos, 1]

    Sometimes the batch dimension will be missing because we applied remove_batch_dim (used when batch_size=1), and we need functions to be robust to that. I THINK I've got everything working, but could easily be wrong!
    """
    def __init__(
        self, 
        cache_dict: dict, 
        model):
        self.cache_dict = cache_dict
        self.model = model
        self.has_batch_dim = True
        self.batch_size = self.cache_dict["hook_embed"].size(0)
        self.ctx_size = self.cache_dict["hook_embed"].size(1)
        
        # Broadcast pos_embed up to batch size, so it has the same shape as all other residual vectors
    
    def remove_batch_dim(self):
        if self.has_batch_dim:
            for key in self.cache_dict:
                assert self.cache_dict[key].size(0)==1, f"Cannot remove batch dimension from cache with batch size > 1, for key {key} with shape {self.cache_dict[key].shape}"
                self.cache_dict[key] = self.cache_dict[key][0]
            self.has_batch_dim = False
        else:
            logging.warning("Tried removing batch dimension after already having removed it.")
    
    def __repr__(self):
        return f"ActivationCache with keys {list(self.cache_dict.keys())}"

    def __getitem__(self, key):
        """ 
        This allows us to treat the activation cache as a dictionary, and do cache["key"] to it. We add bonus functionality to take in shorthand names or tuples - see utils.act_name for the full syntax and examples.

        Dimension order is (act_name, layer_index, layer_type), where layer_type is either "attn" or "mlp" or "ln1" or "ln2" or "ln_final", act_name is the name of the hook (without the hook_ prefix).
        """
        if key in self.cache_dict:
            return self.cache_dict[key]
        elif type(key)==str:
            return self.cache_dict[utils.act_name(key)]
        else:
            if len(key)>1 and key[1] is not None:
                if key[1] < 0:
                    # Supports negative indexing on the layer dimension
                    key = (key[0], self.model.cfg.n_layers+key[1], *key[2:])
            return self.cache_dict[utils.act_name(*key)]
    
    def to(self, device, move_model=False):
        """ 
        Moves the cache to a device - mostly useful for moving it to CPU after model computation finishes to save GPU memory. Matmuls will be much slower on the CPU.

        Note that some methods will break unless the model is also moved to the same device, eg compute_head_results
        """
        self.cache_dict = {key: value.to(device) for key, value in self.cache_dict.items()}

        if move_model:
            self.model.to(device)
    
    def toggle_autodiff(
        self, 
        mode: bool=False
        ):
        """ 
        Sets autodiff to mode (defaults to turning it off). 
        WARNING: This is pretty dangerous, since autodiff is global state - this turns off torch's ability to take gradients completely and it's easy to get a bunch of errors if you don't realise what you're doing.

        But autodiff consumes a LOT of GPU memory (since every intermediate activation is cached until all downstream activations are deleted - this means that computing the loss and storing it in a list will keep every activation sticking around!). So often when you're analysing a model's activations, and don't need to do any training, autodiff is more trouble than its worth.

        If you don't want to mess with global state, using torch.inference_mode as a context manager or decorator achieves similar effects :)
        """
        logging.warning(f"Changed the global state, set autodiff to {mode}")
        torch.set_grad_enabled(mode)
    
    def keys(self):
        return self.cache_dict.keys()
    def values(self):
        return self.cache_dict.values()
    def items(self):
        return self.cache_dict.items()
    
    def accumulated_resid(
        self, 
        layer, 
        incl_mid=False, 
        mlp_input=False,
        return_labels=False):
        """Returns the accumulated residual stream up to a given layer, ie a stack of previous residual streams up to that layer's input. This can be thought of as a series of partial values of the residual stream, where the model gradually accumulates what it wants.

        Args:
            layer (int): The layer to take components up to - by default includes resid_pre for that layer and excludes resid_mid and resid_post for that layer. layer==n_layers means to return all residual streams, including the final one (ie immediately pre logits). The indices are taken such that this gives the accumulated streams up to the input to layer l
            incl_mid (bool, optional): Whether to return resid_mid for all previous layers. Defaults to False.
            mlp_input (bool, optional): Whether to include resid_mid for the current layer - essentially giving MLP input rather than Attn input. Defaults to False.
            return_labels (bool, optional): Whether to return a list of labels for the residual stream components. Useful for labelling graphs. Defaults to True.

        Returns:
            Components: A [num_components, batch_size, pos, d_model] tensor of the accumulated residual streams.
            (labels): An optional list of labels for the components.
        """
        if layer is None or layer==-1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers
        labels = []
        components = []
        for l in range(layer+1):
            if l==self.model.cfg.n_layers:
                components.append(self[('resid_post', self.model.cfg.n_layers-1)])
                labels.append("final_post")
                continue
            components.append(self[("resid_pre", l)])
            labels.append(f"{l}_pre")
            if (incl_mid and l<layer) or (mlp_input and l==layer):
                components.append(self[("resid_mid", l)])
                labels.append(f"{l}_mid")
        
        components = torch.stack(components, dim=0)
        if return_labels:
            return components, labels 
        else:
            return components
    
    def decompose_resid(
        self, 
        layer,
        mlp_input=False,
        mode="all",
        incl_embeds=True,
        return_labels=False):
        """Decomposes the residual stream input to layer L into a stack of the output of previous layers. The sum of these is the input to layer L (plus embedding and pos embedding). This is useful for attributing model behaviour to different components of the residual stream

        Args:
            layer (int): The layer to take components up to - by default includes resid_pre for that layer and excludes resid_mid and resid_post for that layer. layer==n_layers means to return all layer outputs incl in the final layer, layer==0 means just embed and pos_embed. The indices are taken such that this gives the accumulated streams up to the input to layer l
            incl_mid (bool, optional): Whether to return resid_mid for all previous layers. Defaults to False.
            mlp_input (bool, optional): Whether to include attn_out for the current layer - essentially giving MLP input rather than Attn input. Defaults to False.
            mode (str): Values aare "all", "mlp" or "attn". "all" returns all components, "mlp" returns only the MLP components, and "attn" returns only the attention components. Defaults to "all".
            incl_embeds (bool): Whether to include embed & pos_embed
            return_labels (bool, optional): Whether to return a list of labels for the residual stream components. Useful for labelling graphs. Defaults to True.

        Returns:
            Components: A [num_components, batch_size, pos, d_model] tensor of the accumulated residual streams.
            (labels): An optional list of labels for the components.
        """
        if layer is None or layer==-1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers

        incl_attn = mode != "mlp"
        incl_mlp = mode != "attn"
        if incl_embeds:
            components = [self['embed']]
            labels = ['embed']
            if "hook_pos_embed" in self.cache_dict:
                components.append(self['hook_pos_embed'])
                labels.append('pos_embed')
        else:
            components = []
            labels = []

        for l in range(layer):
            if incl_attn:
                components.append(self[("attn_out", l)])
                labels.append(f"{l}_attn_out")
            if incl_mlp:
                components.append(self[("mlp_out", l)])
                labels.append(f"{l}_mlp_out")
        if mlp_input:
            components.append(self[("attn_out", layer)])
            labels.append(f"{layer}_attn_out")
        components = torch.stack(components, dim=0)
        if return_labels:
            return components, labels 
        else:
            return components
    
    def compute_head_results(
        self,
    ):
        """Computes and caches the results for each attention head, ie the amount contributed to the residual stream from that head. attn_out for a layer is the sum of head results plus b_O. Intended use is to enable use_attn_results when running and caching the model, but this can be useful if you forget.
        """
        if 'blocks.0.attn.hook_result' in self.cache_dict:
            logging.warning("Tried to compute head results when they were already cached")
            return
        for l in range(self.model.cfg.n_layers):
            # Note that we haven't enabled set item on this object so we need to edit the underlying cache_dict directly.
            self.cache_dict[f"blocks.{l}.attn.hook_result"] = einsum("... head_index d_head, head_index d_head d_model -> ... head_index d_model", self[("z", l, "attn")], self.model.blocks[l].attn.W_O)
    
    def stack_head_results(
        self,
        layer: int,
        return_labels: bool=False,
        incl_remainder: bool=False,
        pos_slice: Union[Slice, SliceInput]=None,
    ):
        """Returns a stack of all head results (ie residual stream contribution) up to layer L. A good way to decompose the outputs of attention layers into attribution by specific heads.

        Assumes that the model has been run with use_attn_results=True

        Args:
            layer (int): Layer index - heads at all layers strictly before this are included. layer must be in [1, n_layers]
            return_labels (bool, optional): Whether to also return a list of labels of the form "L0H0" for the heads. Defaults to False.
            incl_remainder (bool, optional): Whether to return a final term which is "the rest of the residual stream". Defaults to False.
            pos_slice (Slice): A slice object to apply to the pos dimension. Defaults to None, do nothing.
        """
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        if layer is None or layer==-1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers

        if 'blocks.0.attn.hook_result' not in self.cache_dict:
            raise ValueError("Must run model with use_attn_results=True or run cache.compute_head_results to use this method")

        components = []
        labels = []
        for l in range(layer):
            # Note that this has shape batch x pos x head_index x d_model
            components.append(pos_slice.apply(self[("result", l, "attn")], dim=-3))
            labels.extend([f"L{l}H{h}" for h in range(self.model.cfg.n_heads)])
        if components:
            components = torch.cat(components, dim=-2)
            components = einops.rearrange(components, "... head_index d_model -> head_index ... d_model")
            if incl_remainder:
                remainder = pos_slice.apply(self[("resid_post", layer-1)], dim=-2) - components.sum(dim=0)
                components = torch.cat([components, remainder[None]], dim=0)
                labels.append("remainder")
        elif incl_remainder:
            components = [pos_slice.apply(self[("resid_post", layer-1)], dim=-2)]
        else:
            components = torch.zeros(0, *self["hook_embed"].shape, device=self.model.cfg.device)
            
        if return_labels:
            return components, labels
        else:
            return components
    
    def get_neuron_results(
        self,
        layer: int,
        neuron_slice: Union[Slice, SliceInput] =None,
        pos_slice: Union[Slice, SliceInput] =None,
    ):
        """Returns the results of for neurons in a specific layer (ie, how much each neuron contributes to the residual stream). Does it for the subset of neurons specified by neuron_slice, defaults to all of them. Does *not* cache these because it's expensive in space and cheap to compute.

        Args:
            layer (int): Layer index
            neuron_slice (Slice, optional): Slice of the neuron. Defaults to None.
            pos_slice (Slice, optional): Slice of the positions. Defaults to None. See utils.Slice for details.

        Returns:
            Tensor: [batch_size, pos, d_mlp, d_model] tensor of the results (d_mlp is the neuron index axis)
        """
        if type(neuron_slice) is not Slice:
            neuron_slice = Slice(neuron_slice)
        if type(pos_slice) is not Slice:
            pos_slice = Slice(pos_slice)

        neuron_acts = self[("post", layer, "mlp")]
        W_out = self.model.blocks[layer].mlp.W_out
        if pos_slice is not None:
            # Note - order is important, as Slice.apply *may* collapse a dimension, so this ensures that position dimension is -2 when we apply position slice
            neuron_acts = pos_slice.apply(neuron_acts, dim=-2)
        if neuron_slice is not None:
            neuron_acts = neuron_slice.apply(neuron_acts, dim=-1)
            W_out = neuron_slice.apply(W_out, dim=0)
        return neuron_acts[..., None] * W_out

    def stack_neuron_results(
        self,
        layer: int,
        pos_slice: Union[Slice, SliceInput] =None,
        neuron_slice: Union[Slice, SliceInput] =None,
        return_labels: bool=False,
        incl_remainder: bool=False,
    ):
        """Returns a stack of all neuron results (ie residual stream contribution) up to layer L.

        Args:
            layer (int): Layer index - heads at all layers strictly before this are included. layer must be in [1, n_layers]
            pos_slice (Slice, optional): Slice of the positions. Defaults to None. See utils.Slice for details.
            neuron_slice (Slice, optional): Slice of the neurons. Defaults to None. See utils.Slice for details.
            return_labels (bool, optional): Whether to also return a list of labels of the form "L0H0" for the heads. Defaults to False.
            incl_remainder (bool, optional): Whether to return a final term which is "the rest of the residual stream". Defaults to False.
        """

        if layer is None or layer==-1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers

        components = []
        labels = []

        if not isinstance(neuron_slice, Slice):
            neuron_slice = Slice(neuron_slice)
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)

        neuron_labels = neuron_slice.apply(np.arange(self.model.cfg.d_mlp), dim=0)
        if type(neuron_labels)==int:
            neuron_labels = np.array([neuron_labels])
        for l in range(layer):
            # Note that this has shape batch x pos x head_index x d_model
            components.append(self.get_neuron_results(l, pos_slice=pos_slice, neuron_slice=neuron_slice))
            labels.extend([f"L{l}N{h}" for h in neuron_labels])
        if components:
            components = torch.cat(components, dim=-2)
            components = einops.rearrange(components, "... neuron_index d_model -> neuron_index ... d_model")

            if incl_remainder:
                remainder = self[("resid_post", layer-1)] - components.sum(dim=0)
                components = torch.cat([components, remainder[None]], dim=0)
                labels.append("remainder")
        elif incl_remainder:
            components = [pos_slice.apply(self[("resid_post", layer-1)], dim=-2)]
        else:
            components = torch.zeros(0, *self["hook_embed"].shape, device=self.model.cfg.device)
        if return_labels:
            return components, labels
        else:
            return components
    
    def apply_ln_to_stack(
        self,
        residual_stack: torch.Tensor,
        layer: Optional[int]=None,
        mlp_input: bool=False,
        pos_slice: Union[Slice, SliceInput]  = None,
    ):
        """Takes a stack of components of the residual stream (eg outputs of decompose_resid or accumulated_resid), treats them as the input to a specific layer, and applies the layer norm scaling of that layer to them, using the cached scale factors.

        Args:
            residual_stack (torch.Tensor): A tensor, whose final dimension is d_model. The other trailing dimensions are assumed to be the same as the stored hook_scale - which may or may not include batch or position dimensions.
            layer (int): The layer we're taking the input to. In [0, n_layers], n_layers means the unembed. None maps to the n_layers case, ie the unembed.
            mlp_input (bool, optional): Whether the input is to the MLP or attn (ie ln2 vs ln1). Defaults to False, ie ln1. If layer==n_layers, must be False, and we use ln_final
            pos_slice: The slice to take of positions, if residual_stack is not over the full context, None means do nothing. See utils.Slice for details. Defaults to None.
        """
        # First, center
        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        if layer is None or layer==-1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers
        residual_stack = residual_stack - residual_stack.mean(dim=-1, keepdim=True)

        if layer==self.model.cfg.n_layers or layer is None:
            scale = self['ln_final.hook_scale']
        else:
            hook_name = f"blocks.{layer}.ln{2 if mlp_input else 1}.hook_scale"
            scale = self[hook_name]

        # The shape of scale is [batch, position, 1] - final dimension is a dummy thing to get broadcoasting to work nicely.
        scale = pos_slice.apply(scale, dim=-2)
        
        return residual_stack / scale

    def get_full_resid_decomposition(
        self,
        layer: Optional[int]=None,
        mlp_input=False,
        apply_ln=True,
        pos_slice: Union[Slice, SliceInput]  = None,
        return_labels=False,
    ):
        """Returns the full decomposition of the residual stream into embed, pos_embed, each head result, each neuron result, and the accumulated biases. We break down the residual stream that is input into some layer.

        Args:
            layer (int): The layer we're inputting into. layer is in [0, n_layers], if layer==n_layers (or None) we're inputting into the unembed (the entire stream), if layer==0 then it's just embed and pos_embed
            mlp_input (bool, optional): Are we inputting to the MLP in that layer or the attn? Must be False for final layer, since that's the unembed. Defaults to False.
            apply_ln (bool, optional): Whether to apply LayerNorm to the stack. Defaults to True.
            pos_slice (Slice, optional): Slice of the positions to take. Defaults to None. See utils.Slice for details.
            return_labels (bool): Whether to return the labels. Defaults to False.
        """
        if layer is None or layer==-1:
            # Default to the residual stream immediately pre unembed
            layer = self.model.cfg.n_layers

        if not isinstance(pos_slice, Slice):
            pos_slice = Slice(pos_slice)
        head_stack, head_labels = self.stack_head_results(layer + (1 if mlp_input else 0), pos_slice=pos_slice, return_labels=True)
        neuron_stack, neuron_labels = self.stack_neuron_results(layer, pos_slice=pos_slice, return_labels=True)
        bias = self.model.accumulated_bias(layer, mlp_input)
        if self.has_batch_dim:
            bias = einops.repeat(bias, "d_model -> batch ctx d_model", batch=self.batch_size, ctx=self.ctx_size)
        else:
            bias = einops.repeat(bias, "d_model -> ctx d_model", ctx=self.ctx_size)

        labels = head_labels + neuron_labels + ["embed", "pos_embed", "bias"]
        embed = pos_slice.apply(self["embed"], -2)[None]
        pos_embed = pos_slice.apply(self["pos_embed"], -2)[None]
        bias = pos_slice.apply(bias, -2)[None]
        l = [head_stack, neuron_stack, embed, pos_embed, bias]
        for i in l: print(i.shape)
        residual_stack = torch.cat(l, dim=0)

        if apply_ln:
            residual_stack = self.apply_ln_to_stack(residual_stack, layer, pos_slice=pos_slice)

        if return_labels:
            return residual_stack, labels
        else:
            return residual_stack

