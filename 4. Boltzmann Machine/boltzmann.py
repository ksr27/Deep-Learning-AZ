#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 12:54:25 2017

@author: user
"""

# Import libs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Import dataset
movies=pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')