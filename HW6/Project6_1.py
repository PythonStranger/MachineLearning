#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 10:31:43 2018

@author: tianxiang
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# data from 1/1/2009 - 8/1/2018
td = pd.read_csv('PAYNSA-3.csv')

# convert to numpy
n = np.shape(td)[0]
T = np.arange(0, n)
y = (np.delete(td.values, [0], axis=1)).astype(float)

plt.plot(y)
plt.title('RAW DATA')
plt.show()

model = LinearRegression()
model.fit(T.reshape(n, 1), y)

# get the line 
predictions = model.predict(T[:, np.newaxis])

# Detrend the y data
detrended_y = (y - predictions).ravel()
plt.plot(detrended_y)
plt.title('DETRENDED DATA')
plt.show()

# Decompose the signal using FFT
result = np.fft.fft(detrended_y)

half_n = int(n / 2)
result = result[range(half_n)]
freqs = np.arange(half_n)  # frequencies 0 - 60
As = np.abs(result) / half_n  # amplitudes
phis = np.angle(result) * 180 / np.pi + 90

plt.plot(freqs, As)
plt.title('FREQUENCY Domain')
plt.show()

tops = np.where(As > 800, 1, 0)  # tops show what is frequency
top_As = tops * As
top_phis = tops * phis

# Convert back to domain
j = 0
new_y = np.zeros(n)
a = []
for i in tops:
    new_y += top_As[j] * np.sin(2 * np.pi * (j * int(i)) * T + top_phis[j] * (np.pi / 180))
    j += 1

plt.plot(detrended_y)
plt.plot(new_y)
plt.title('FREQUENCY-BASED MODEL')
plt.show()

residuals = detrended_y.ravel() - new_y
plt.plot(residuals)

plt.title('RESIDUALS')
plt.show()
plt.hist(residuals, bins=20)
plt.title('RESIDUALS HISTOGRAM')
plt.show()
