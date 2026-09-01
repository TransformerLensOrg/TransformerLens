"""Exception types a runtime type-check rejection can surface as.

Which library wraps the violation is a dependency detail that has already moved
once: jaxtyping <0.3 let beartype's ``BeartypeCallHintParamViolation`` propagate,
while >=0.3 re-raises its own ``TypeCheckError`` around it. Tests here assert that
invalid input *is rejected*, not which wrapper reports it, so they should accept
any of these — otherwise the next bump breaks a handful of tests that were never
about the wrapper.
"""

from beartype.roar import BeartypeCallHintParamViolation
from jaxtyping import TypeCheckError as JaxtypingTypeCheckError
from typeguard import TypeCheckError as TypeguardTypeCheckError

TYPECHECK_ERRORS = (
    BeartypeCallHintParamViolation,
    JaxtypingTypeCheckError,
    TypeguardTypeCheckError,
)
