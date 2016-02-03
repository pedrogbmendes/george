#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import itertools

sys.path.append("..")

import george
from george.kernels import TaskKernel


num_tasks = 10


kernel = TaskKernel(1,0,num_tasks)

print(kernel.vector)
kernel.vector=range(1, len(kernel.vector)+1)
print(kernel.vector)


K = np.zeros([num_tasks, num_tasks])

for (i,j) in itertools.product(range(num_tasks), repeat=2):
	K[i,j] = (kernel.value(np.array([[i]]),np.array([[j]]))[0,0])

print(K)


print(np.linalg.cholesky(K))
