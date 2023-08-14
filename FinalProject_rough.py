# -*- coding: utf-8 -*-
"""
2023-05-02

PHY408 Final Report Rough Work
"""
import numpy as np
import matplotlib.pyplot as plt

# Load data
dat = np.loadtxt("unemployment_data.csv", delimiter = ",", skiprows = 1, usecols = (1, 2, 3), unpack = True)
month = np.loadtxt("unemployment_data.csv", dtype = str, delimiter = ",", skiprows = 1, usecols = 0, unpack = True)
cpi = dat[0]
wages = dat[1]
unemploy = dat[2]

# Inflation adjust
cpi_2001 = cpi / 96.3 * 100
adj_wages = wages / cpi_2001 * 100

# # Unadjusted plot
# plt.figure()
# plt.plot(month, adj_wages, label = "Real wages ($CAD)")
# plt.plot(month, unemploy, label = "Youth unemployment rate (%)")
# plt.xlabel("Month")
# plt.title("Real Wages and Youth Unemployment, 2001 - Present")
# plt.legend()
# plt.xticks(month[::12], rotation = 45)

# Detrend
num_month = np.arange(len(month))
wages_trend = np.polyfit(num_month, adj_wages, 1)
wages_detrend = adj_wages - np.polyval(wages_trend, num_month)
unemploy_trend = np.polyfit(num_month, unemploy, 6)
unemploy_detrend = unemploy - np.polyval(unemploy_trend, num_month)

# # Detrended plot
# plt.figure()
# plt.plot(month, adj_wages, label = "Real wages ($CAD)")
# plt.plot(month, wages_detrend, label = "Real wages ($CAD) (Detrended)")
# plt.plot(month, unemploy, label = "Youth unemployment rate (%)")
# plt.plot(month, unemploy_detrend, label = "Youth unemployment rate (%) (Detrended)")
# plt.xlabel("Month")
# plt.title("Real Wages and Youth Unemployment, 2001 - Present")
# plt.legend()
# plt.xticks(month[::12], rotation = 45)

# F-domain filter
wages_ft = np.fft.fft(wages_detrend)
unemploy_ft = np.fft.fft(unemploy_detrend)
f = np.fft.fftfreq(len(wages_detrend), 1/12)

wages_ft_an = wages_ft
wages_ft_an[(f > 0.9) | (f < -0.9)] = 0
unemploy_ft_an = unemploy_ft
unemploy_ft_an[(f > 0.9) | (f < -0.9)] = 0

wages_filter = np.real(np.fft.ifft(wages_ft_an))
unemploy_filter = np.real(np.fft.ifft(unemploy_ft_an))

wages_retrend = wages_filter + np.polyval(wages_trend, num_month)
unemploy_retrend = unemploy_filter + np.polyval(unemploy_trend, num_month)

# Filter plot
plt.figure()
plt.plot(month, adj_wages, label = "Real wages ($CAD)")
plt.plot(month, wages_retrend, label = "Real wages ($CAD) (Retrended)")
plt.plot(month, unemploy, label = "Youth unemployment rate (%)")
plt.plot(month, unemploy_retrend, label = "Youth unemployment rate (%) (Retrended)")
plt.xlabel("Month")
plt.title("Real Wages and Youth Unemployment, Original vs Filtered Data")
plt.legend()
plt.xticks(month[::12], rotation = 45)

# Cross-correlation
cross_corr = np.fft.fftshift(np.fft.ifft(wages_ft * np.conj(unemploy_ft)))
t = np.arange(-len(wages_ft)/2, len(wages_ft)/2)

# Cross-corr plot
plt.figure()
plt.plot(t, np.real(cross_corr))