from typing import Dict, Optional

from .generic import GenericImplementation
from .newman.pytorch import PytorchImplementation as NewmanPytorchImplementation
from .sbm.numpy import NumpyImplementation as SBMNumpyImplementation
from .sbm.pytorch import PytorchImplementation as SBMPytorchImplementation
from .sbm.pytorch import PytorchLogImplementation as SBMPytorchLogImplementation

SBM_IMPLEMENTATIONS: Dict[str, GenericImplementation] = {
    "numpy": SBMNumpyImplementation(),
    "pytorch": SBMPytorchImplementation(),
    "pytorch_log": SBMPytorchLogImplementation(),
}

NEWMAN_IMPLEMENTATIONS: Dict[str, GenericImplementation] = {
    "pytorch": NewmanPytorchImplementation(),
}


def get_implementation(
    implementation: str, model: Optional[str] = "sbm"
) -> GenericImplementation:
    try:
        if model == "sbm":
            return SBM_IMPLEMENTATIONS[implementation]
        elif model == "newman":
            return NEWMAN_IMPLEMENTATIONS[implementation]
        else:
            raise ValueError(f"Unknown model {model}.")
    except KeyError:
        raise ValueError(f"Unknown implementation {implementation} for model {model}.")
