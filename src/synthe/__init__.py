"""Top-level package for synthe."""

__author__ = """T. Moudiki"""
__email__ = 'thierry.moudiki@gmail.com'

from .conformal_inference import ConformalInference  # noqa: F401
from .empirical_copula import EmpiricalCopula  # noqa: F401 

__all__ = [
    "ConformalInference",
    "EmpiricalCopula",
]