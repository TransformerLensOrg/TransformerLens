"""Compatibility shims for transformers version differences.

These patches are applied lazily (only when missing) so they're safe to call
from multiple adapters — the first caller wins, subsequent calls are no-ops.

WARNING: patches here mutate classes from the installed `transformers` package
in place. They are process-global and persist for the entire Python session —
every model loaded afterward, including ones unrelated to the caller, sees the
patched class. This is acceptable because the shims only *add* v4-era methods
that v5 removed; they do not change v5 behavior. But it means a bug in a shim
affects the whole session, not just the adapter that invoked it.

REMOVAL: drop the corresponding block (and its call sites) once the minimum
supported `transformers` version provides the method natively, or once all
remote-code models we support have been updated for v5. Track upstream status
against `transformers.cache_utils.DynamicCache` — when `from_legacy_cache`,
`to_legacy_cache`, and `get_usable_length` are restored or no longer needed,
`patch_dynamic_cache_v5` can be deleted outright.
"""


def patch_dynamic_cache_v5() -> None:
    """Backfill DynamicCache methods removed in transformers v5.

    Remote-code models written for transformers v4 call from_legacy_cache,
    to_legacy_cache, and get_usable_length which were removed in v5.
    Call this from any adapter's prepare_loading() that needs them.

    Side effect: mutates `transformers.cache_utils.DynamicCache` for the whole
    process. See module docstring.
    """
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return

    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod  # type: ignore[misc]
        def _from_legacy_cache(cls, past_key_values=None):  # type: ignore[no-untyped-def]
            cache = cls()
            if past_key_values is not None:
                for idx, layer_past in enumerate(past_key_values):
                    cache.update(layer_past[0], layer_past[1], idx)
            return cache

        DynamicCache.from_legacy_cache = _from_legacy_cache  # type: ignore[attr-defined]

    if not hasattr(DynamicCache, "get_usable_length"):

        def _get_usable_length(self, new_seq_len: int = 0, layer_idx: int = 0) -> int:  # type: ignore[no-untyped-def]
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]

    if not hasattr(DynamicCache, "to_legacy_cache"):

        def _to_legacy_cache(self):  # type: ignore[no-untyped-def]
            return tuple((layer.keys, layer.values) for layer in self.layers)

        DynamicCache.to_legacy_cache = _to_legacy_cache  # type: ignore[attr-defined]
