"""
DiffusionModel Examples - Converting Colab notebook examples
"""

import numpy as np
import matplotlib.pyplot as plt
from synthe import SyntheticTabularSampler

synthtabular = SyntheticTabularSampler(type="classification")

data = synthtabular.sample()

print(data)

synthtabular = SyntheticTabularSampler(type="regression")

data = synthtabular.sample()

print(data)