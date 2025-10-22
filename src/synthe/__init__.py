"""Top-level package for synthe."""

__author__ = """T. Moudiki"""
__email__ = "thierry.moudiki@gmail.com"

from .adaptivehistsampler import AdaptiveHistogramSampler  # noqa: F401
from .distro_simulator import DistroSimulator  # noqa: F401
from .empirical_copula import EmpiricalCopula  # noqa: F401
from .stratified_sampling import StratifiedClusteringSubsampling
from .row_subsampling import SubSampler
from .healthsims import SmartHealthSimulator  # noqa: F401
from .metrics import DistanceMetrics  # noqa: F401
from .meboot import MaximumEntropyBootstrap

__all__ = [
    "AdaptiveHistogramSampler",
    "DistroSimulator",
    "EmpiricalCopula",
    "StratifiedClusteringSubsampling",
    "SubSampler",
    "SmartHealthSimulator",
    "DistanceMetrics",
    "MaximumEntropyBootstrap",
]
