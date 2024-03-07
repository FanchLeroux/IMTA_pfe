# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:54:58 2024     under Python 3.11.7

@author: f24lerou
"""

# 8<---------------------------- Add path --------------------------------

import os
import sys

path = os.path.abspath(os.path.abspath('..'))
sys.path.append(path)

# 8<--------------------------- Import modules ---------------------------

import numpy as np

from tools import getCartesianCoordinates 

# 8<------------------------- Functions definitions ----------------------


def gaussian(size_support, side_length, waist, distance):
    return