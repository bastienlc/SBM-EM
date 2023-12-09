from typing import Dict

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
}
