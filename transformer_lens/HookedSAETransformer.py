import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from jaxtyping import Float

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint  # Hooking utilities
from transformer_lens.HookedSAE import HookedSAE
from transformer_lens.HookedTransformer import HookedTransformer

SingleLoss = Float[torch.Tensor, ""]  # Type alias for a single element tensor
LossPerToken = Float[torch.Tensor, "batch pos-1"]
Loss = Union[SingleLoss, LossPerToken]


def get_deep_attr(obj: Any, path: str):
    """Helper function to get a nested attribute from a object.
    In practice used to access HookedTransformer HookPoints (eg model.blocks[0].attn.hook_z)

    Args:
        obj: Any object. In practice, this is a HookedTransformer (or subclass)
        path: str. The path to the attribute you want to access. (eg "blocks.0.attn.hook_z")

    returns:
        Any. The attribute at the end of the path
    """
    parts = path.split(".")
    # Navigate to the last component in the path
    for part in parts:
        if part.isdigit():  # This is a list index
            obj = obj[int(part)]
        else:  # This is an attribute
            obj = getattr(obj, part)
    return obj


def set_deep_attr(obj: Any, path: str, value: Any):
    """Helper function to change the value of a nested attribute from a object.
    In practice used to swap HookedTransformer HookPoints (eg model.blocks[0].attn.hook_z) with HookedSAEs

    Args:
        obj: Any object. In practice, this is a HookedTransformer (or subclass)
        path: str. The path to the attribute you want to access. (eg "blocks.0.attn.hook_z")
        value: Any. The value you want to set the attribute to (eg a HookedSAE object)
    """
    parts = path.split(".")
    # Navigate to the last component in the path
    for part in parts[:-1]:
        if part.isdigit():  # This is a list index
            obj = obj[int(part)]
        else:  # This is an attribute
            obj = getattr(obj, part)
    # Set the value on the final attribute
    setattr(obj, parts[-1], value)


class HookedSAETransformer(HookedTransformer):
    def __init__(
        self,
        *model_args,
        **model_kwargs,
    ):
        """Model initialization. Just HookedTransformer init, but adds a dictionary to attach SAEs.

        Note that if you want to load the model from pretrained weights, you should use
        :meth:`from_pretrained` instead.

        Args:
            *model_args: Positional arguments for HookedTransformer initialization
            **model_kwargs: Keyword arguments for HookedTransformer initialization
        """
        super().__init__(*model_args, **model_kwargs)
        self.acts_to_saes: Dict[str, HookedSAE] = {}

    def attach_sae(self, sae: HookedSAE, turn_on: bool = True, hook_name: Optional[str] = None):
        """Attach an SAE to the model.

        By default, it will use the hook_name from the SAE's HookedSAEConfig. If you want to use a different hook_name, you can pass it in as an argument.
        By default, the SAE will be turned on. If you want to attach the SAE without turning it on, you can pass in turn_on=False.

        Args:
            sae: (HookedAutoEncoder) SAE that you want to attach
            turn_on: if true, turn on the SAE (default: True)
            hook_name: (Optional[str]) The hook name to attach the SAE to (default: None)
        """
        act_name = hook_name or sae.cfg.hook_name
        if (act_name not in self.acts_to_saes) and (act_name not in self.hook_dict):
            logging.warning(
                f"No hook found for {act_name}. Skipping. Check model.hook_dict for available hooks."
            )
            return
        if act_name in self.acts_to_saes:
            logging.warning(f"SAE already attached to {act_name}. This will be replaced.")
        self.acts_to_saes[act_name] = sae
        if turn_on:
            self.turn_saes_on([act_name])

    def turn_saes_on(self, act_names: Optional[Union[str, List[str]]] = None):
        """
        Turn on the attached SAEs for the given act_name(s)

        Note they will stay on you turn them off

        Args:
            act_names: (Union[str, List[str]]) The act_name(s) for the SAEs to turn on. If None, will turn on all SAEs attached to the model. Defaults to None.
        """
        if isinstance(act_names, str):
            act_names = [act_names]

        for act_name in act_names or self.acts_to_saes.keys():
            if act_name not in self.acts_to_saes:
                logging.warning(f"No SAE is attached to {act_name}. Skipping.")
            else:
                set_deep_attr(self, act_name, self.acts_to_saes[act_name])

        self.setup()

    def turn_saes_off(self, act_names: Optional[Union[str, List[str]]] = None):
        """
        Turns off the SAEs for the given act_name(s)

        If no act_names are given, will turn off all SAEs

        Args:
            act_names: (Optional[Union[str, List[str]]]) The act_names for the SAEs to turn off. Defaults to None.
        """
        if isinstance(act_names, str):
            act_names = [act_names]

        for act_name in act_names or self.acts_to_saes.keys():
            if act_name not in self.acts_to_saes:
                logging.warning(f"No SAE is attached to {act_name}. There's nothing to turn off.")
            else:
                set_deep_attr(self, act_name, HookPoint())

        self.setup()

    def run_with_saes(
        self,
        *model_args,
        act_names: Union[str, List[str]] = [],
        reset_saes_end: bool = True,
        **model_kwargs,
    ) -> Union[
        None,
        Float[torch.Tensor, "batch pos d_vocab"],
        Loss,
        Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
    ]:
        """Wrapper around HookedTransformer forward pass.

        Runs the model with the SAEs for given act_names turned on.
        Note this will turn off all other SAEs that are not in act_names before running
        After running, it will turn off all SAEs

        Args:
            *model_args: Positional arguments for the model forward pass
            act_names: (Union[str, List[str]]) The act_names for the SAEs to turn on for this forward pass
            reset_saes_end (bool): If True, removes all SAEs added during this run are turned off at the end. Default is True.
            **model_kwargs: Keyword arguments for the model forward pass
        """
        self.turn_saes_off()
        with self.saes(act_names=act_names, reset_saes_end=reset_saes_end):
            return self(*model_args, **model_kwargs)

    def run_with_cache_with_saes(
        self,
        *model_args,
        act_names: Union[str, List[str]] = [],
        reset_saes_end: bool = True,
        return_cache_object=True,
        remove_batch_dim=False,
        **kwargs,
    ) -> Tuple[
        Union[
            None,
            Float[torch.Tensor, "batch pos d_vocab"],
            Loss,
            Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
        ],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """Wrapper around 'run_with_cache' in HookedTransformer.

        Turns on the SAEs for the given act_names before running the model with cache and then turns them off after
        Note this will turn off all other SAEs that are not in act_names before running
        After running, it will turn off all SAEs

        Args:
            *model_args: Positional arguments for the model forward pass
            act_names: (Union[str, List[str]]) The act_names for the SAEs to turn on for this forward pass
            reset_saes_end: (bool) Whether to turn off the SAEs corresponding to act_names at the end of the forward pass (default: True)
            return_cache_object: (bool) if True, this will return an ActivationCache object, with a bunch of
                useful HookedTransformer specific methods, otherwise it will return a dictionary of
                activations as in HookedRootModule.
            remove_batch_dim: (bool) Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
            **kwargs: Keyword arguments for the model forward pass
        """
        self.turn_saes_off()
        with self.saes(act_names=act_names, reset_saes_end=reset_saes_end):
            return self.run_with_cache(
                *model_args,
                return_cache_object=return_cache_object,
                remove_batch_dim=remove_batch_dim,
                **kwargs,
            )

    def run_with_hooks_with_saes(
        self,
        *model_args,
        act_names: Union[str, List[str]] = [],
        reset_saes_end: bool = True,
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end=True,
        clear_contexts=False,
        **model_kwargs,
    ):
        """Wrapper around 'run_with_hooks' in HookedTransformer.

        Turns on the SAEs for the given act_names before running the model with hooks and then turns them off after
        Note this will turn off all other SAEs that are not in act_names before running
        After running, it will turn off all SAEs

        ARgs:
            *model_args: Positional arguments for the model forward pass
            act_names: (Union[str, List[str]]) The act_names for the SAEs to turn on for this forward pass
            reset_saes_end: (bool) Whether to turn off the SAEs corresponding to act_names at the end of the forward pass (default: True)
            fwd_hooks: (List[Tuple[Union[str, Callable], Callable]]) List of forward hooks to apply
            bwd_hooks: (List[Tuple[Union[str, Callable], Callable]]) List of backward hooks to apply
            reset_hooks_end: (bool) Whether to reset the hooks at the end of the forward pass (default: True)
            clear_contexts: (bool) Whether to clear the contexts at the end of the forward pass (default: False)
            **model_kwargs: Keyword arguments for the model forward pass
        """
        self.turn_saes_off()
        with self.saes(act_names=act_names, reset_saes_end=reset_saes_end):
            return self.run_with_hooks(
                *model_args,
                fwd_hooks=fwd_hooks,
                bwd_hooks=bwd_hooks,
                reset_hooks_end=reset_hooks_end,
                clear_contexts=clear_contexts,
                **model_kwargs,
            )

    def get_saes_status(self):
        """
        Helper function to check which SAEs attached to the model are currently turned on / off

        Returns:
            Dict[str, bool]: A dictionary of act_name to whether the corresponding SAE is turned on
        """
        return {
            act_name: (False if isinstance(get_deep_attr(self, act_name), HookPoint) else True)
            for act_name in self.acts_to_saes.keys()
        }

    def remove_all_saes(self):
        """
        Turns off and removes all SAEs attached to the model
        """
        self.turn_saes_off()
        self.acts_to_saes = {}

    @contextmanager
    def saes(
        self,
        act_names: Union[str, List[str]] = [],
        reset_saes_end: bool = True,
    ):
        """
        A context manager for adding temporary SAEs to the model.
        See HookedTransformer.hooks for a similar context manager for hooks.

        Args:
            act_names: hook point names where SAEs should be spliced.
            reset_saes_end (bool): If True, removes all SAEs added by this context manager when the context manager exits.

        Example:

        .. code-block:: python

            with model.saes(act_names=["blocks.10.hook_resid_pre"]):
                spliced_loss = model(text, return_type="loss")
        """
        try:
            self.turn_saes_on(act_names)
            yield self
        finally:
            if reset_saes_end:
                self.turn_saes_off(act_names)
