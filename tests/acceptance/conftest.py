"""Session fixtures for acceptance tests.

transformer_lens imports stay inside fixture bodies — jaxtyping's pytest_configure
hook must install before the package is first imported.
"""

import pytest
