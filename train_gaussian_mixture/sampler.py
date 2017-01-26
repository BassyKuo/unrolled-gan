# -*- coding: utf-8 -*-
import math
import numpy as np

def gaussian_mixture(batchsize, num_cluster=8, scale=1, std=1):
	rand_indices = np.random.randint(0, num_cluster, size=batchsize)
	base_angle = math.pi * 2 / num_cluster
	angle = rand_indices * base_angle
	mean = np.zeros((batchsize, 2), dtype=np.float32)
	mean[:, 0] = np.cos(angle) * scale
	mean[:, 1] = np.sin(angle) * scale
	return np.random.normal(mean, std**2, (batchsize, 2)).astype(np.float32)