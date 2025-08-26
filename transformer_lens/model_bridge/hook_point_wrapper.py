"""Hook point wrapper for providing dotted access to hook points."""

from transformer_lens.hook_points import HookPoint


class HookPointWrapper:
    """Wrapper class to provide dotted access to hook points."""

    def __init__(self, hook_in: HookPoint, hook_out: HookPoint):
        """Initialize the wrapper with hook_in and hook_out points.

        Args:
            hook_in: The input hook point
            hook_out: The output hook point
        """
        self.hook_in = hook_in
        self.hook_out = hook_out
