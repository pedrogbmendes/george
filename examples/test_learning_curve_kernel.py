#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
sys.path.append("..")

import george
from george.kernels import LearningCurveKernel


kernel = LearningCurveKernel(ndim=1,dim=0)
kernel.vector=np.array([1.0, 0.5])

n_curves = 10

gp = george.GP(kernel, mean=0)

t = np.linspace(0, 9, 100)

for i in range(20):
    plt.plot(gp.sample(t))
plt.show()

for i in range(n_curves):
	x = np.array(random.sample(range(10), 3))

	alpha = np.random.rand()
	print(alpha)
	y = np.exp(-x * alpha) + np.random.randn() * 0.001
	gp.compute(x[:, None])

	f = gp.predict(y, t[:, None])[0]
	plt.plot(f)
plt.show()
