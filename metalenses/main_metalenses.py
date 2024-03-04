# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:19:09 2024

@author: f24lerou
"""

import numpy as np
import matplotlib.pyplot as plt

from doe.phaseMasks import lens, tilt

plt.imshow(tilt(deltaPhi=2*np.pi, n_levels=3))

