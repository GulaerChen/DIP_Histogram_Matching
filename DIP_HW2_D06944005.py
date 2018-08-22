# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 20:06:13 2017

@author: Gulaer
"""

import numpy as np
import cv2

origin = cv2.imread('Hakodate.jpg')

"""Histogram Equalization"""

equ = np.copy(origin)

for i  in xrange (3) :
    equ[:,:,i] = cv2.equalizeHist(origin[:,:,i])
    
cv2.imwrite('Hakodate_equ.jpg',equ)

"""Histogram Matching"""

benchmark =  cv2.imread('Hakodate_tem.jpg')
matched = np.copy(origin)

def hist_match(origin, benchmark):

    origin_shape = origin.shape
    
    origin = origin.ravel()
    benchmark = benchmark.ravel()

    o_values, bin_idx, o_counts = np.unique(origin, return_inverse=True,return_counts=True)
    b_values, b_counts = np.unique(benchmark, return_counts=True)

    o_quantiles = np.cumsum(o_counts).astype(np.float64)
    o_quantiles /= o_quantiles[-1]
    b_quantiles = np.cumsum(b_counts).astype(np.float64)
    b_quantiles /= b_quantiles[-1]

    interp_t_values = np.interp(o_quantiles, b_quantiles, b_values)

    return interp_t_values[bin_idx].reshape(origin_shape)

for i  in xrange (3) :
    matched[:,:,i] = hist_match(origin[:,:,i], benchmark[:,:,i])

cv2.imwrite('Hakodate_matched.jpg', matched)