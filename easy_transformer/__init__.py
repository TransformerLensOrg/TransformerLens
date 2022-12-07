"""Alias for TransformerLens

EasyTransformer has been renamed to TransformerLens. This is an alias package so
that the change is non-breaking. However you should instead import
`transformer_lens`."""
import sys
from warnings import warn

# Make sure transformer_lens is in sys.modules
import transformer_lens

# Warn the user that this has been renamed
warn("easy_transformer has been renamed to transformer_lens")

# Link this module to transformer_lens
sys.modules[__name__] = sys.modules['transformer_lens']
