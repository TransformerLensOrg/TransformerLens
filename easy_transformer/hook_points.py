import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
from typing_extensions import Literal

HookDirection = Literal["forward", "backward"]
FORWARD_HOOK_DIRECTION = "forward"
BACKWARD_HOOK_DIRECTION = "backward"
BOTH_HOOK_DIRECTION = "both"


class NamesFilter(ABC):
    @abstractmethod
    def is_included(self, name: str) -> bool:
        pass


class FunctionalNameFilter(NamesFilter):
    def __init__(self, f: Callable[[str], bool]):
        self.f = f

    def is_included(self, name: str) -> bool:
        return self.f(name)


class ConstantNameFilter(NamesFilter):
    def __init__(self, names: Sequence[str]):
        self.names = set(names)

    def is_included(self, name: str) -> bool:
        return name in self.names

class SingleNameFilter(NamesFilter):
    def __init__(self, name: str):
        self.name = name

    def is_included(self, name: str) -> bool:
        return self.name == name

# %%


class HookPoint(nn.Module):
    """
    A helper class to get access to intermediate activations.

    It's a dummy module that is the identity function by default
    I can wrap any intermediate activation in a HookPoint and get a convenient
    way to add PyTorch hooks.

    It's inspired by Garcon from Anthropic: https://transformer-circuits.pub/2021/garcon/index.html

    See [EasyTransformer_Demo.ipynb] for an example of how to use it.
    """

    def __init__(self):
        super().__init__()
        # TODO make sure the experiments still run after this rename- probalby just change it in the experiments
        self.forward_hooks = []
        self.backward_hooks = []
        self.context = {}

        # The name of the hook from the perspective of the root
        # module. This is set by the root module at setup.
        self.name: str

    def add_hook(self, hook, direction: HookDirection = FORWARD_HOOK_DIRECTION):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output,
        # which are the same for a HookPoint)

        if direction == FORWARD_HOOK_DIRECTION:

            def full_hook(module, module_input, module_output):
                return hook(module_output, hook=self)

            handle = self.register_forward_hook(full_hook)
            self.forward_hooks.append(handle)
        # TODO find a tool that complains when i try to test two different-typed things for equality

        elif direction == BACKWARD_HOOK_DIRECTION:

            def full_hook(module, module_input, module_output):
                # For a backwards hook, module_output is a tuple of (grad,) - I don't know why.
                grad = module_output[0]
                return hook(grad, hook=self)

            handle = self.register_full_backward_hook(full_hook)
            self.backward_hooks.append(handle)
        else:
            # TODO for neel nanda- do we want to allow [BOTH_HOOK_DIRECTION]?
            raise ValueError(f"Invalid direction {direction}")

    def remove_hooks(self, direction: HookDirection = FORWARD_HOOK_DIRECTION):
        def remove_forward_hooks():
            for handle in self.forward_hooks:
                handle.remove()
            self.forward_hooks = []

        def remove_backward_hooks():
            for handle in self.backward_hooks:
                handle.remove()
            self.backward_hooks = []

        if direction == FORWARD_HOOK_DIRECTION:
            remove_forward_hooks()
        elif direction == BACKWARD_HOOK_DIRECTION:
            remove_backward_hooks()
        elif direction == BOTH_HOOK_DIRECTION:
            remove_forward_hooks()
            remove_backward_hooks()
        else:
            raise ValueError(f"Invalid direction {direction}")

    def clear_context(self):
        del self.context
        self.context = {}

    def forward(self, x):
        """Identity function"""
        return x

    def layer(self):
        """
        Returns the layer index if the name has the form 'blocks.{layer}.{...}'
        Helper function that's mainly useful on EasyTransformer
        If it doesn't have this form, raises an error -
        """
        try:
            layer = int(self.name.split(".")[1])
            return int(layer)
        except ValueError:
            # TODO maybe return [None] instead of raising an error? but test it first, might
            # break in subtle ways
            raise ValueError(
                f"HookPoint name {self.name} doesn't have the form 'blocks.layer.rest"
            )


# %%
# TODO maybe this should be a decorator so clients don't have to call setup()?
# or just a class should be enough?
class HookedRootModule(nn.Module):
    """
    A class building on nn.Module to interface nicely with HookPoints
    Adds various nice utilities, most notably run_with_hooks to run the model with temporary hooks, and run_with_cache to run the model on some input and return a cache of all activations

    See [EasyTransformer_Demo.ipynb] for an example of how to use it.

    WARNING: The main footgun with PyTorch hooking is that hooks are GLOBAL state. If you add a hook to the module, and then run it a bunch of times, the hooks persist. If you debug a broken hook and add the fixed version, the broken one is still there. To solve this, run_with_hooks will remove hooks at the start and end by default, and I recommend using the API of this and run_with_cache. If you want to add hooks into global state, I recommend being intentional about this, and I recommend using reset_hooks liberally in your code to remove any accidentally remaining global state.

    The main time this goes wrong is when you want to use backward hooks (to cache or intervene on gradients). In this case, you need to keep the hooks around as global state until you've run loss.backward() (and so need to disable the reset_hooks_end flag on run_with_hooks)
    """

    def __init__(self):
        super().__init__()
        self.is_caching = False  # see [run_with_hooks]  and [add_caching_hooks]

    def setup(self):
        """
        Setup function - this needs to be run in the model's [__init__]
        AFTER defining all layers.
        """
        # Add a parameter to each module giving its name.
        self.name_to_module = {}
        # Build a dictionary mapping a module name to the module
        self.name_to_hook = {}
        for name, module in self.named_modules():
            module.name = name
            self.name_to_module[
                name
            ] = module  # we don't use [self.named_modules()] because that's a generator, not a dictionary
            # this is actually the cleanest way to do this 
            if "HookPoint" in str(type(module)):
                self.name_to_hook[name] = module

    def hook_points(self):
        return self.name_to_hook.values()

    def remove_all_hook_fns(self, direction=BOTH_HOOK_DIRECTION):
        for hp in self.hook_points():
            hp.remove_hooks(direction)

    def clear_contexts(self):
        for hp in self.hook_points():
            hp.clear_context()

    def reset_hooks(self, clear_contexts=True, direction=BOTH_HOOK_DIRECTION):
        if clear_contexts:
            self.clear_contexts()
        self.remove_all_hook_fns(direction)
        self.is_caching = False

    def add_hook(self, name, hook, direction: HookDirection = FORWARD_HOOK_DIRECTION):
        if type(name) == str:
            self.name_to_module[name].add_hook(hook, direction=direction)
        else:
            # Otherwise, name is a Boolean function on names
            for hook_name, hp in self.name_to_hook.items():
                if name(hook_name):
                    hp.add_hook(hook, direction=direction)

    def __apply_hooks__(self, hooks, direction: HookDirection):
        for name, hook in hooks:
            if type(name) == str:
                self.name_to_module[name].add_hook(hook, direction=direction)
            else:
                # TODO add a class here
                # Otherwise, name is a Boolean function on names
                for hook_name, hp in self.name_to_hook.items():
                    if name(hook_name):
                        hp.add_hook(hook, direction=direction)

    def run_with_hooks(
        self,
        *model_args,
        forward_hooks=[],
        backward_hooks=[],
        reset_hooks_start=True,
        reset_hooks_end=True,
        clear_contexts=False,
        **model_kwargs,
    ):
        """
        forward_hooks: A list of (name, hook), where name is either the name of
        a hook point or a Boolean function on hook names and hook is the
        function to add to that hook point, or the hook whose names evaluate
        to True respectively. Ditto bwd_hooks
        reset_hooks_end (bool): If True, all hooks are removed at the end (ie,
        including those added in this run)
        clear_contexts (bool): If True, clears hook contexts whenever hooks are reset
        Note that if we want to use backward hooks, we need to set
        reset_hooks_end to be False, so the backward hooks are still there - this function only runs a forward pass.
        """
        if reset_hooks_start:
            if self.is_caching:
                logging.warning("Caching is on, but hooks are being reset")
            self.reset_hooks(clear_contexts)
        self.__apply_hooks__(forward_hooks, FORWARD_HOOK_DIRECTION)
        self.__apply_hooks__(backward_hooks, BACKWARD_HOOK_DIRECTION)
        out = self.forward(*model_args, **model_kwargs)
        if reset_hooks_end:
            if len(backward_hooks) > 0:
                logging.warning(
                    "WARNING: Hooks were reset at the end of run_with_hooks while backward hooks were set. This removes the backward hooks before a backward pass can occur"
                )
            self.reset_hooks(clear_contexts)
        return out

    def add_caching_hooks(
        self,
        names_filter: Optional[NamesFilter] = None,
        include_backward: bool = False,
        device=None,
        remove_batch_dim: bool = False,
        cache: Optional[dict] = None,
    ) -> dict:
        """Adds hooks to the model to cache activations. Note: It does NOT actually run the model to get activations, that must be done separately.

        Args:
            names_filter (NamesFilter, optional): Which activations to cache. Can be a list of strings (hook names) or a filter function mapping hook names to booleans. Defaults to lambda name: True.
            include_backward (bool, optional): Whether to also do backwards hooks. Defaults to False.
            device (_type_, optional): The device to store on. Defaults to CUDA if available else CPU.
            remove_batch_dim (bool, optional): Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
            cache (Optional[dict], optional): The cache to store activations in, a new dict is created by default. Defaults to None.

        Returns:
            cache (dict): The cache where activations will be stored.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if cache is None:
            cache = {}

        if names_filter is None:
            names_filter = FunctionalNameFilter(lambda _: True)

        self.is_caching = True

        def save_hook(tensor, hook):
            if remove_batch_dim:
                cache[hook.name] = tensor.detach().to(device)[0]
            else:
                cache[hook.name] = tensor.detach().to(device)

        def save_hook_back(tensor, hook):
            if remove_batch_dim:
                cache[hook.name + "_grad"] = tensor.detach().to(device)[0]
            else:
                cache[hook.name + "_grad"] = tensor.detach().to(device)

        for name, hp in self.name_to_hook.items():
            if names_filter.is_included(name):
                hp.add_hook(save_hook, FORWARD_HOOK_DIRECTION)
                if include_backward:
                    hp.add_hook(save_hook_back, BACKWARD_HOOK_DIRECTION)
        return cache

    def run_with_cache(
        self,
        *model_args,
        names_filter: Optional[NamesFilter] = None,
        device=None,
        remove_batch_dim=False,
        include_backward=False,
        reset_hooks_end=True,
        clear_contexts=False,
        **model_kwargs,
    ):
        """
        Runs the model and returns model output and a Cache object

        model_args and model_kwargs - all positional arguments and keyword arguments not otherwise captured are input to the model
        names_filter (None or str or [str] or fn:str->bool): a filter for which activations to cache. Defaults to None, which means cache everything.
        device (str or torch.Device): The device to cache activations on, defaults to model device. Note that this must be set if the model does not have a model.cfg.device attribute. WARNING: Setting a different device than the one used by the model leads to significant performance degradation.
        remove_batch_dim (bool): If True, will remove the batch dimension when caching. Only makes sense with batch_size=1 inputs.
        include_backward (bool): If True, will call backward on the model output and also cache gradients. It is assumed that the model outputs a scalar, ie. return_type="loss", for predict the next token loss. Custom loss functions are not supported
        reset_hooks_start (bool): If True, all prior hooks are removed at the start
        reset_hooks_end (bool): If True, all hooks are removed at the end (ie,
        including those added in this run)
        clear_contexts (bool): If True, clears hook contexts whenever hooks are reset
        """
        cache_dict = self.add_caching_hooks(
            names_filter=names_filter,
            include_backward=include_backward,
            device=device,
            remove_batch_dim=remove_batch_dim,
        )
        model_out = self(*model_args, **model_kwargs)

        if include_backward:
            model_out.backward()

        if reset_hooks_end:
            self.reset_hooks(clear_contexts)
        return model_out, cache_dict

    def cache_all(
        self, cache, include_backward=False, device=None, remove_batch_dim=False
    ):
        logging.warning(
            "cache_all is deprecated and will eventually be removed, use add_caching_hooks or run_with_cache"
        )
        self.add_caching_hooks(
            names_filter=FunctionalNameFilter(lambda _: True),
            cache=cache,
            include_backward=include_backward,
            device=device,
            remove_batch_dim=remove_batch_dim,
        )

    def cache_some(
        self,
        cache,
        names: Callable[[str], bool],
        include_backward=False,
        device=None,
        remove_batch_dim=False,
    ):
        """Cache a list of hook provided by names, Boolean function on names"""
        logging.warning(
            "cache_some is deprecated and will eventually be removed, use add_caching_hooks or run_with_cache"
        )
        self.add_caching_hooks(
            names_filter=FunctionalNameFilter(names),
            cache=cache,
            include_backward=include_backward,
            device=device,
            remove_batch_dim=remove_batch_dim,
        )
