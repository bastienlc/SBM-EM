from typing import Dict

from src.Newman_EM.implementations.newman_torch import Newman_pytorch_implementation

from .generic import GenericImplementation
from .numpy import NumpyImplementation
from .pytorch import PytorchImplementation, PytorchLogImplementation

IMPLEMENTATIONS: Dict[str, GenericImplementation] = {
    "numpy": NumpyImplementation(),
    "pytorch": PytorchImplementation(),
    "pytorch_log": PytorchLogImplementation(),
    "newman_pytorch": Newman_pytorch_implementation(),
}


def get_implementation(implementation: str) -> GenericImplementation:
    return IMPLEMENTATIONS[implementation]
