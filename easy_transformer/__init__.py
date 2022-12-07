import sys

# make sure bar is in sys.modules
import transformer_lens

# link this module to bar
sys.modules[__name__] = sys.modules['transformer_lens']

# Or simply
# sys.modules[__name__] = __import__('transformer_lens')