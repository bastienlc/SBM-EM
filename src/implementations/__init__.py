from typing import Dict

from src.Newman_EM.implementations.newman_torch import Newman_pytorch_implementation

from .generic import GenericImplementation
from .numpy import NumpyImplementation
from .python import PythonImplementation
from .pytorch import (
    PytorchImplementation,
    PytorchLogImplementation,
    PytorchLowMemoryImplementation,
)

IMPLEMENTATIONS: Dict[str, GenericImplementation] = {
    "numpy": NumpyImplementation(),
    "python": PythonImplementation(),
    "pytorch": PytorchImplementation(),
    "pytorch_low_memory": PytorchLowMemoryImplementation(),
    "pytorch_log": PytorchLogImplementation(),
    "newman_pytorch": Newman_pytorch_implementation(),
}


def get_implementation(implementation: str) -> GenericImplementation:
    return IMPLEMENTATIONS[implementation]
