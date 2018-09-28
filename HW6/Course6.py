#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 10:15:12 2018

@author: tianxiang
"""

import matplotlib.pyplot as plt
import numpy as np
n = 100
period = 1
sampling_rate = 1 / 100
t = np.arange( 0, period, sampling_rate )
# The first sine wave
A1 = 2
f1 = 3
phi1 = 0 * ( np.pi / 180 )
y1 = A1 * np.sin( 2 * np.pi * f1 * t + phi1 )
# The second sine wave
A2 = 1
f2 = 10
phi2 = 5 * ( np.pi / 180 )
y2 = A2 * np.sin( 2 * np.pi * f2 * t + phi2 )
# The sum of the two sine waves
y = y1 + y2

#plt.plot( t, y)
#plt.show()

# Let's decompose the signal using fast Fourier transform
result = np.fft.fft( y )


# The return values are complex conjugates of complex numbers
# So, we only need to get one side, or half, of them
half_n = int( n / 2 )

result = result[ range( half_n ) ]
freqs = np.arange( half_n ) # frequencies 0 - 50
plt.plot(freqs, np.abs(result))
plt.show()
As = np.abs( result ) / half_n # Amplitudes
# We want the amplitude, regardless of sign
# So, we take the abs()
plt.plot( freqs, As )
plt.show()