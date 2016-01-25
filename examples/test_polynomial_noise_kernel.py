#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import itertools

import matplotlib.pyplot as plt

sys.path.append("..")
import george
from george.kernels import HeteroscedasticNoisePolynomialKernel


kernel = HeteroscedasticNoisePolynomialKernel(2,0)





num_plot_points=100
x = np.linspace(0,1, num_plot_points)
K = np.zeros(num_plot_points,dtype=np.double)

for foo in range(7):
	kernel.vector=np.random.randn(len(kernel.vector))

	for i,s in enumerate(x):
		K[i] = kernel.value(np.array([[s,0]]),np.array([[s,0]]))[0,0]

	plt.plot(x,K, label="{} -> c={:.2f}, alpha = {:.2f}".format(kernel.vector, np.exp(kernel.vector[0]), np.exp(kernel.vector[1])), linewidth=3)
	plt.legend()

plt.xlabel('input value')
plt.ylabel('kernel value')
plt.title("Note how the parameters of the kernel live in log space!")
plt.show()
