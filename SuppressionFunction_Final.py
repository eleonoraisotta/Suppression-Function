# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 13:59:02 2023

@author: EIsotta
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad
from pathlib import Path

#plotting parameters
colors = sns.color_palette('deep') #'cubehelix' for greyscale
# 0 blue, 1 orange, 2 green, 3 red, 4 purple, 5 brown, 6 pink, 7 grey, 8 sand, 9 cyan 
linestyles = [(0, ()),  # 0 solid
              (0, (1, 10)),  # 1 'loosely dotted'
              (0, (1, 5)),  # 2 dotted
              (0, (1, 1)),  # 3 'densely dotted'
              (0, (5, 10)),  # 4 'loosely dashed'
              (0, (5, 5)),  # 5 'dashed'
              (0, (5, 1)),  # 6 'densely dashed'
              (0, (3, 10, 1, 10)),  # 7 'loosely dashdotted'
              (0, (3, 5, 1, 5)),  # 8'dashdotted'
              (0, (3, 1, 1, 1)),  # 9 'densely dashdotted'
              (0, (3, 10, 1, 10, 1, 10)),  # 10 'loosely dashdotdotted'
              (0, (3, 5, 1, 5, 1, 5)),  # 11'dashdotdotted'
              (0, (3, 1, 1, 1, 1, 1))]  # 12 'densely dashdotdotted'
lw = 1.5 # linewidth
ms = 6 # markersize
ts = 16 #textsize
sns.set_style('whitegrid', {'axes.edgecolor':'black'})
sns.set_context('notebook') # default 'notebook', or 'paper', 'talk', 'poster'

# destination path
output_path = Path("./") # this will save to the folder where you saved the code. Please change accordingly

 
# %%
''' GENERAL DEFINITIONS'''

# Calculation of Silicon kappa(T) according to Debye-Callaway model
# see Phonon engineering through crystal chemistry, DOI: 10.1039/c1jm11754h

# Global Constants
h = 6.62607015*1e-34 # m2 kg / s
hbar = 1.054571817*1e-34 # m2 kg / s or J s
kB = 1.380649*1e-23 # m2 kg s-2 K-1
Na = 6.02214076e23 # Avogadro

# Material Constants - SILICON
N = 8 # number of atmos per unit cell
V = 1/N*161.32e-30 # Volume per atom [m**3]
v_Si = 5830   # experimental speed of sound for single crystal
gamma = 1.0 # Grueneisen parameter, DFT mode averaged
Mav = 0.0280855/Na # average atomic mass [kg]
wmax = (6*np.pi**2/V)**(1/3)*v_Si # maximum frequency - Debye model


# Relevant arrays
T = np.linspace(300, 700, num=5) # temperature [K]
t = np.linspace(0.00, 1.00, num=11) # transmission probability
x = np.logspace(-12, -2, num=100) # distance from grain boundary [m]
g = np.linspace(2.4, 26.8, num = 3) # g parameter in trasmission probability, from Hori, Shiomi, Dames, App. Phys. Lett., 2015

# %%
''' GENERAL FUNCTIONS'''

def kappa_Umklapp(T):
    kappa_U_HT = ((6*np.pi**2)**(2/3))/(4*np.pi**2)*Mav*v_Si**3/(T*V**(2/3)*gamma**2)  # W/(mK)
    return kappa_U_HT

def MFP_Umklapp(w, T):
    MFP = v_Si**4 * (6 * np.pi**2)**(1/3) * Mav / (2 * kB * gamma**2 * w**2 * T * V**(1/3))
    return MFP

def SpecHeat_HT(w):
    C = 3*kB*w**2/(2*np.pi**2*v_Si**3)
    return C

def SpecHeat_Debye(w, T):
    C = 3*hbar**2/(2*np.pi**2*kB*T**2) * (w**4*np.exp(hbar*w/kB/T)) / (v_Si**3*(np.exp(hbar*w/kB/T)-1)**2)
    return C
    
def kappa_U_spectral(w, T, Debye = True):
    if Debye:
        C = SpecHeat_Debye(w, T)
    else:
        C = SpecHeat_HT(w)
    MFP = MFP_Umklapp(w, T)
    kappas = 1/3 * C * v_Si * MFP
    return kappas

def transmission(w, g): # transmission probability, from Hori, Shiomi, Dames, App. Phys. Lett., 2015
    t = 1 / ( g * w / wmax + 1)
    return t

# %%  
''' PLOT - kappa versus temperature'''

kappa_U_Debye = np.zeros(len(T))
for i, Ti in enumerate(T):
    kappa_U_Debye[i], _ = quad(lambda w: kappa_U_spectral(w, Ti), 0, wmax)
    
fig, (ax) = plt.subplots(figsize=(5.3,4))
ax.set_ylabel('$\u03BA$(T) [W/(mK)]', fontsize=ts) 
ax.set_xlabel('T [K]', fontsize=ts) 
ax.tick_params(axis="both", labelsize=ts-1)
ax.grid(False)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.set_ylim([48, 152])

ax.plot(T, kappa_U_Debye, linewidth=lw*2, color='k', label='Umklapp scattering\nDebye Cp')

ax.legend(loc='best', frameon=False, fontsize=ts)

fig.tight_layout()
name_of_figure = 'kappa_T_silicon'
plt.savefig(output_path / name_of_figure, dpi=500, bbox_inches='tight')

# %%    
''' DEFINITION - SUPPRESSION FUNCTIONS '''

d = np.linspace(0, 100, num=10000)  # d = x/MFP, aka the inverse of the Knudsen number


# Suppression Function - All Heat Flux components (sections 2.1-2.3 in the paper)

def SuppFunc_Iso(d, t): #t = transmission probability
    return np.where(d <= 1, ( (1-t) * np.log(1/d) + 1/d * (1+t) + 1 - t) / (2 / d), 1)


# Suppression Function - Perpendicular Heat Flux component only (section 2.4 paper)

def SuppFunc_X(d, t): #  t = transmission probability
    return np.where(d <= 1, 3/2 * (1-t) * ( d - d**3 ) + t * ( 1 - d**3 ) +  d**3 , 1)

F_iso = np.zeros((len(d), len(t)))
F_X = np.zeros((len(d), len(t)))

for i in range(len(t)):
    F_iso[:, i] = SuppFunc_Iso(d, t[i])
    F_X[:, i] = SuppFunc_X(d, t[i])

# %% 
''' PLOT - Suppression function isotropic'''

fig, (ax) = plt.subplots(1, 1, figsize=(5.3,4))

ax.set_ylabel('F(x)', fontsize=ts) 
ax.set_xlabel('$x/\Lambda_{bulk}$', fontsize=ts) 
ax.tick_params(axis="both", labelsize=ts-1)
ax.grid(False)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.set_ylim([-0.05, 1.05])
ax.set_xscale('log')

ax.plot(d, F_iso[:,10], linewidth=lw*2, color=colors[3], label='t = 1')
ax.plot(d, F_iso[:,8], linewidth=lw, color='grey')
ax.plot(d, F_iso[:,6], linewidth=lw, color='grey')
ax.plot(d, F_iso[:,4], linewidth=lw, color='grey')
ax.plot(d, F_iso[:,2], linewidth=lw, color='grey')
ax.plot(d, F_iso[:,0], linewidth=lw*2, color='k', label='t = 0')
ax.legend(loc='best', frameon=False, fontsize=ts)
ax.set_xticks([0.01, 0.1, 1, 10, 100])
ax.set_xticklabels(['0.01', '0.1', '1', '10', '100'])

fig.tight_layout()
name_of_figure = 'SuppFunc_Iso_transmission'
plt.savefig(output_path / name_of_figure, dpi=500, bbox_inches='tight')

# %% 
''' PLOT - Suppression function X component'''

fig, (ax) = plt.subplots(1, 1, figsize=(5.3,4))

#ax.set_ylabel('$\Lambda(x)$ / $\Lambda_{bulk}$', fontsize=ts) 
ax.set_ylabel('F(x)', fontsize=ts) 
ax.set_xlabel('$x/\Lambda_{bulk}$', fontsize=ts) 
ax.tick_params(axis="both", labelsize=ts-1)
ax.grid(False)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.set_ylim([-0.05, 1.05])
ax.set_xscale('log')

ax.plot(d, F_X[:,10], linewidth=lw*2, color=colors[3], label='t = 1')
ax.plot(d, F_X[:,8], linewidth=lw, color='grey')
ax.plot(d, F_X[:,6], linewidth=lw, color='grey')
ax.plot(d, F_X[:,4], linewidth=lw, color='grey')
ax.plot(d, F_X[:,2], linewidth=lw, color='grey')
ax.plot(d, F_X[:,0], linewidth=lw*2, color='k', label='t = 0')
ax.legend(loc='best', frameon=False, fontsize=ts)
ax.set_xticks([0.01, 0.1, 1, 10, 100])
ax.set_xticklabels(['0.01', '0.1', '1', '10', '100'])

fig.tight_layout()
name_of_figure = 'SuppFunc_X_transmission'
plt.savefig(output_path / name_of_figure, dpi=500, bbox_inches='tight')

# %%  
''' DEFINITION - SPECTRAL THERMAL CONDUCTIVITY (x) '''

# Spectral thermal conductivity as a function of distance - Isotropic Supp Func, constant transmission, Debye Cp

def kappa_spectral_Fiso_tconst(w, x_i, t, T=300, Debye=True):
    if Debye: C = SpecHeat_Debye(w, T) 
    else: C = SpecHeat_HT(w)
    MFP = MFP_Umklapp(w, T)
    F_x = SuppFunc_Iso(x_i/MFP, t)
    kappasx = 1/3 * C * v_Si * MFP * F_x
    return  kappasx 

# Spectral thermal conductivity as a function of distance x - Isotropic Supp Func, frequency-dependent transmission, Debye Cp

def kappa_spectral_Fiso_tg(w, x_i, g, T=300, Debye=True):
    if Debye: C = SpecHeat_Debye(w, T)  
    else: C = SpecHeat_HT(w)
    MFP = MFP_Umklapp(w, T)
    t = transmission(w, g)
    F_x = SuppFunc_Iso(x_i/MFP, t)
    kappasx = 1/3 * C * v_Si * MFP * F_x
    return  kappasx 


# Spectral thermal conductivity as a function of distance - X-projected Supp Func, constant transmission, Debye Cp

def kappa_spectral_FX_tconst(w, x_i, t, T=300, Debye=True):
    if Debye: C = SpecHeat_Debye(w, T)  
    else: C = SpecHeat_HT(w)
    MFP = MFP_Umklapp(w, T)
    F_x = SuppFunc_X(x_i/MFP, t)
    kappasx = 1/3 * C * v_Si * MFP * F_x
    return  kappasx 

# Spectral thermal conductivity as a function of distance - X-direction Supp Func, frequency-dependent transmission, Debye Cp

def kappa_spectral_FX_tg(w, x_i, g, T=300, Debye=True):
    if Debye: C = SpecHeat_Debye(w, T)  
    else: C = SpecHeat_HT(w)
    MFP = MFP_Umklapp(w, T)
    t = transmission(w, g)
    F_x = SuppFunc_X(x_i/MFP, t)
    kappasx = 1/3 * C * v_Si * MFP * F_x
    return  kappasx 


# %% 
''' CALCULATION - THERMAL CONDUCTIVITY PROFILE, no transmission '''

# Thermal conductivity as a function of distance - Isotropic Supp Func, no transmission, Debye Cp
kappa_Fiso_t0 = np.zeros(len(x))

# Thermal conductivity as a function of distance - X projected Supp Func, no transmission, Debye Cp
kappa_FX_t0 = np.zeros(len(x))

for i in range(len(x)):
    kappa_Fiso_t0[i], _ = quad(lambda w: kappa_spectral_Fiso_tconst(w, x[i], 0), 0, wmax)
    kappa_FX_t0[i], _ = quad(lambda w: kappa_spectral_FX_tconst(w, x[i], 0), 0, wmax)
        
# %% 
''' PLOT - Thermal conductivity profile isotropic, no transmission '''

fig, (ax) = plt.subplots(1, 1, figsize=(5.3,4))

ax.set_ylabel('$\u03BA$(x) [W/(mK)]', fontsize=ts) 
ax.set_xlabel('Distance from GB [$m$]', fontsize=ts) 
ax.tick_params(axis="both", labelsize=ts-1)
ax.grid(False)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.set_xlim([1e-10, 1e-2])
ax.set_ylim([-5, 155])
ax.set_xscale('log')

ax.plot(x, kappa_Fiso_t0, linewidth=lw*2, color='k', label='Isotropic, t = 0')

ax.legend(loc='best', frameon=False, fontsize=ts)

fig.tight_layout()
name_of_figure = 'Kappa_Iso_t0'
plt.savefig(output_path / name_of_figure, dpi=500, bbox_inches='tight')     
   
# %% 
''' PLOT - Thermal conductivity profile X-component, no transmission '''

fig, (ax) = plt.subplots(1, 1, figsize=(5.3,4))

ax.set_ylabel('$\u03BA$(x) [W/(mK)]', fontsize=ts) 
ax.set_xlabel('Distance from GB [$m$]', fontsize=ts) 
ax.tick_params(axis="both", labelsize=ts-1)
ax.grid(False)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.set_xlim([1e-10, 1e-2])
ax.set_ylim([-5, 155])
ax.set_xscale('log')

ax.plot(x, kappa_FX_t0, linewidth=lw*2, color='k', label='X component, t = 0')

ax.legend(loc='best', frameon=False, fontsize=ts)

fig.tight_layout()
name_of_figure = 'Kappa_X_t0'
plt.savefig(output_path / name_of_figure, dpi=500, bbox_inches='tight')     
   
# %% 
''' CALCULATION - THERMAL CONDUCTIVITY PROFILE, transmission '''

# Thermal conductivity as a function of distance - Isotropic projected Supp Func, frequency-dependent transmission, Temperature-dependent
kappa_Fiso_tg = np.zeros((len(x), len(g), len(T))) # Debye Cp

# Thermal conductivity as a function of distance - X projected Supp Func, frequency-dependent transmission, Temperature-dependent
kappa_FX_tg = np.zeros((len(x), len(g), len(T))) # Debye Cp

for k in range(len(T)):
    for j in range(len(g)):
        for i in range(len(x)):
            kappa_FX_tg[i,j,k], _ = quad(lambda w: kappa_spectral_FX_tg(w, x[i], g[j], T[k]), 0, wmax)
            kappa_Fiso_tg[i,j,k], _ = quad(lambda w: kappa_spectral_Fiso_tg(w, x[i], g[j], T[k]), 0, wmax)
                   
# %% 

''' PLOT - Thermal conductivity profile isotropic, transmission '''

fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(16,4))

ax.set_ylabel('$\u03BA$(x) [W/(mK)]', fontsize=ts) 
ax.set_xlabel('Distance from GB [$m$]', fontsize=ts) 
ax.set_title('$\gamma$ = 2.4', fontsize=ts) 
ax.tick_params(axis="both", labelsize=ts-1)
ax.grid(False)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.set_xlim([1e-10, 1e-2])
ax.set_ylim([-5, 165])
ax.set_xscale('log')

ax.plot(x, kappa_Fiso_tg[:,0,0], linewidth=lw*2, color=colors[0], label='Iso, 300K')
ax.plot(x, kappa_Fiso_tg[:,0,1], linewidth=lw*2, color=colors[1], label='Iso, 400K')
ax.plot(x, kappa_Fiso_tg[:,0,2], linewidth=lw*2, color=colors[2], label='Iso, 500K')
ax.plot(x, kappa_Fiso_tg[:,0,3], linewidth=lw*2, color=colors[3], label='Iso, 600K')
ax.plot(x, kappa_Fiso_tg[:,0,4], linewidth=lw*2, color=colors[4], label='Iso, 700K')
#ax.legend(loc='best', frameon=False, fontsize=ts)

ax1.set_ylabel('$\u03BA$(x) [W/(mK)]', fontsize=ts) 
ax1.set_xlabel('Distance from GB [$m$]', fontsize=ts) 
ax1.set_title('$\gamma$ = 14.6', fontsize=ts) 
ax1.tick_params(axis="both", labelsize=ts-1)
ax1.grid(False)
ax1.xaxis.tick_bottom()
ax1.yaxis.tick_left()
ax1.set_xlim([1e-10, 1e-2])
ax1.set_ylim([-5, 165])
ax1.set_xscale('log')

ax1.plot(x, kappa_Fiso_tg[:,1,0], linewidth=lw*2, color=colors[0], label='Iso, 300K')
ax1.plot(x, kappa_Fiso_tg[:,1,1], linewidth=lw*2, color=colors[1], label='Iso, 400K')
ax1.plot(x, kappa_Fiso_tg[:,1,2], linewidth=lw*2, color=colors[2], label='Iso, 500K')
ax1.plot(x, kappa_Fiso_tg[:,1,3], linewidth=lw*2, color=colors[3], label='Iso, 600K')
ax1.plot(x, kappa_Fiso_tg[:,1,4], linewidth=lw*2, color=colors[4], label='Iso, 700K')
#ax1.legend(loc='best', frameon=False, fontsize=ts)

ax2.set_ylabel('$\u03BA$(x) [W/(mK)]', fontsize=ts) 
ax2.set_xlabel('Distance from GB [$m$]', fontsize=ts)
ax2.set_title('$\gamma$ = 26.8', fontsize=ts)  
ax2.tick_params(axis="both", labelsize=ts-1)
ax2.grid(False)
ax2.xaxis.tick_bottom()
ax2.yaxis.tick_left()
ax2.set_xlim([1e-10, 1e-2])
ax2.set_ylim([-5, 165])
ax2.set_xscale('log')

ax2.plot(x, kappa_Fiso_tg[:,2,0], linewidth=lw*2, color=colors[0], label='Iso, 300K')
ax2.plot(x, kappa_Fiso_tg[:,2,1], linewidth=lw*2, color=colors[1], label='Iso, 400K')
ax2.plot(x, kappa_Fiso_tg[:,2,2], linewidth=lw*2, color=colors[2], label='Iso, 500K')
ax2.plot(x, kappa_Fiso_tg[:,2,3], linewidth=lw*2, color=colors[3], label='Iso, 600K')
ax2.plot(x, kappa_Fiso_tg[:,2,4], linewidth=lw*2, color=colors[4], label='Iso, 700K')
ax2.legend(loc='best', frameon=False, fontsize=ts, bbox_to_anchor=(1.05, 1))

fig.tight_layout()
name_of_figure = 'Kappax_tg_Iso_Temperature'
plt.savefig(output_path / name_of_figure, dpi=500, bbox_inches='tight')

# %%

''' PLOT - Thermal conductivity profile x-component, transmission '''

fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(16,4))

ax.set_ylabel('$\u03BA$(x) [W/(mK)]', fontsize=ts) 
ax.set_xlabel('Distance from GB [$m$]', fontsize=ts) 
ax.set_title('$\gamma$ = 2.4', fontsize=ts) 
ax.tick_params(axis="both", labelsize=ts-1)
ax.grid(False)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.set_xlim([1e-10, 1e-2])
ax.set_ylim([-5, 155])
ax.set_xscale('log')

ax.plot(x, kappa_FX_tg[:,0,0], linewidth=lw*2, color=colors[0], label='X comp., 300K')
ax.plot(x, kappa_FX_tg[:,0,1], linewidth=lw*2, color=colors[1], label='X comp., 400K')
ax.plot(x, kappa_FX_tg[:,0,2], linewidth=lw*2, color=colors[2], label='X comp., 500K')
ax.plot(x, kappa_FX_tg[:,0,3], linewidth=lw*2, color=colors[3], label='X comp., 600K')
ax.plot(x, kappa_FX_tg[:,0,4], linewidth=lw*2, color=colors[4], label='X comp., 700K')
#ax.legend(loc='best', frameon=False, fontsize=ts)

ax1.set_ylabel('$\u03BA$(x) [W/(mK)]', fontsize=ts) 
ax1.set_xlabel('Distance from GB [$m$]', fontsize=ts) 
ax1.set_title('$\gamma$ = 14.6', fontsize=ts) 
ax1.tick_params(axis="both", labelsize=ts-1)
ax1.grid(False)
ax1.xaxis.tick_bottom()
ax1.yaxis.tick_left()
ax1.set_xlim([1e-10, 1e-2])
ax1.set_ylim([-5, 155])
ax1.set_xscale('log')

ax1.plot(x, kappa_FX_tg[:,1,0], linewidth=lw*2, color=colors[0], label='X comp., 300K')
ax1.plot(x, kappa_FX_tg[:,1,1], linewidth=lw*2, color=colors[1], label='X comp., 400K')
ax1.plot(x, kappa_FX_tg[:,1,2], linewidth=lw*2, color=colors[2], label='X comp., 500K')
ax1.plot(x, kappa_FX_tg[:,1,3], linewidth=lw*2, color=colors[3], label='X comp., 600K')
ax1.plot(x, kappa_FX_tg[:,1,4], linewidth=lw*2, color=colors[4], label='X comp., 700K')
#ax1.legend(loc='best', frameon=False, fontsize=ts)

ax2.set_ylabel('$\u03BA$(x) [W/(mK)]', fontsize=ts) 
ax2.set_xlabel('Distance from GB [$m$]', fontsize=ts)
ax2.set_title('$\gamma$ = 26.8', fontsize=ts)  
ax2.tick_params(axis="both", labelsize=ts-1)
ax2.grid(False)
ax2.xaxis.tick_bottom()
ax2.yaxis.tick_left()
ax2.set_xlim([1e-10, 1e-2])
ax2.set_ylim([-5, 155])
ax2.set_xscale('log')

ax2.plot(x, kappa_FX_tg[:,2,0], linewidth=lw*2, color=colors[0], label='X comp., 300K')
ax2.plot(x, kappa_FX_tg[:,2,1], linewidth=lw*2, color=colors[1], label='X comp., 400K')
ax2.plot(x, kappa_FX_tg[:,2,2], linewidth=lw*2, color=colors[2], label='X comp., 500K')
ax2.plot(x, kappa_FX_tg[:,2,3], linewidth=lw*2, color=colors[3], label='X comp., 600K')
ax2.plot(x, kappa_FX_tg[:,2,4], linewidth=lw*2, color=colors[4], label='X comp., 700K')
ax2.legend(loc='best', frameon=False, fontsize=ts, bbox_to_anchor=(1.05, 1))

fig.tight_layout()
name_of_figure = 'Kappax_tg_X_Debye_Temperature'
plt.savefig(output_path / name_of_figure, dpi=500, bbox_inches='tight')

# %% 
''' DEFINITION - EXCESS RESISTIVITY (x), transmission '''

# Integration limits for the inner integral (over frequency)
w_min = 0 
w_max = wmax

# Excess resistivity - F isotropic

def excess_resistivity_Fiso_tg(x, g, T=300, Debye=True):
    # Calculate the inner integral for the current x
    inner_integral, _ = quad(lambda w: kappa_spectral_Fiso_tg(w, x, g, T, Debye), w_min, w_max)
        
    if Debye: k_bulk, _ = quad(lambda w: kappa_U_spectral(w, T, Debye), w_min, w_max)
    else: k_bulk = kappa_Umklapp(T)
    # Calculate the integrand value for the current x
    integrand_value = 2 * (1/inner_integral - 1/k_bulk) # 2 is needed for the two sides of the boundary
    return integrand_value


# Excess resistivity - F X-direction

def excess_resistivity_FX_tg(x, g, T=300, Debye=True):
    # Calculate the inner integral for the current x
    inner_integral, _ = quad(lambda w: kappa_spectral_FX_tg(w, x, g, T, Debye), w_min, w_max)
        
    if Debye: k_bulk, _ = quad(lambda w: kappa_U_spectral(w, T, Debye), w_min, w_max)
    else: k_bulk = kappa_Umklapp(T)
    # Calculate the integrand value for the current x
    integrand_value = 2 * (1/inner_integral - 1/k_bulk)
    return integrand_value

# %% 

''' CALCULATION - EXCESS RESISTIVITY (x), transmission '''

# Excess resistivity as a function of distance - Isotropic projected Supp Func, frequency-dependent transmission, Temperature-dependent
ExRes_Fiso_tg_D = np.zeros((len(x), len(g), len(T))) # Debye Cp

# Excess resistivity as a function of distance - X projected Supp Func, frequency-dependent transmission, Temperature-dependent
ExRes_FX_tg_D = np.zeros((len(x), len(g), len(T))) # Debye Cp

for k in range(len(T)):
    for j in range(len(g)):
        for i in range(len(x)):
            ExRes_Fiso_tg_D[i,j,k] = excess_resistivity_Fiso_tg(x[i], g[j], T[k])
            ExRes_FX_tg_D[i,j,k] = excess_resistivity_FX_tg(x[i], g[j], T[k])
            

# %% 

''' PLOT - Excess resistivity profile isotropic, transmission '''

fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(16,4))

ax.set_ylabel('$\u03BA$(x) [W/(mK)]', fontsize=ts) 
ax.set_xlabel('Distance from GB [$m$]', fontsize=ts) 
ax.set_title('$\gamma$ = 2.4', fontsize=ts) 
ax.tick_params(axis="both", labelsize=ts-1)
ax.grid(False)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.set_xlim([1e-10, 1e-2])
ax.set_ylim([-0.0005, 0.023])
ax.set_xscale('log')

ax.plot(x, ExRes_Fiso_tg_D[:,0,0], linewidth=lw*2, color=colors[0], label='Iso, 300K')
ax.plot(x, ExRes_Fiso_tg_D[:,0,1], linewidth=lw*2, color=colors[1], label='Iso, 400K')
ax.plot(x, ExRes_Fiso_tg_D[:,0,2], linewidth=lw*2, color=colors[2], label='Iso, 500K')
ax.plot(x, ExRes_Fiso_tg_D[:,0,3], linewidth=lw*2, color=colors[3], label='Iso, 600K')
ax.plot(x, ExRes_Fiso_tg_D[:,0,4], linewidth=lw*2, color=colors[4], label='Iso, 700K')
#ax.legend(loc='best', frameon=False, fontsize=ts)

ax1.set_ylabel('$\u03BA$(x) [W/(mK)]', fontsize=ts) 
ax1.set_xlabel('Distance from GB [$m$]', fontsize=ts) 
ax1.set_title('$\gamma$ = 14.6', fontsize=ts) 
ax1.tick_params(axis="both", labelsize=ts-1)
ax1.grid(False)
ax1.xaxis.tick_bottom()
ax1.yaxis.tick_left()
ax1.set_xlim([1e-10, 1e-2])
ax1.set_ylim([-0.0005, 0.023])
ax1.set_xscale('log')

ax1.plot(x, ExRes_Fiso_tg_D[:,1,0], linewidth=lw*2, color=colors[0], label='Iso, 300K')
ax1.plot(x, ExRes_Fiso_tg_D[:,1,1], linewidth=lw*2, color=colors[1], label='Iso, 400K')
ax1.plot(x, ExRes_Fiso_tg_D[:,1,2], linewidth=lw*2, color=colors[2], label='Iso, 500K')
ax1.plot(x, ExRes_Fiso_tg_D[:,1,3], linewidth=lw*2, color=colors[3], label='Iso, 600K')
ax1.plot(x, ExRes_Fiso_tg_D[:,1,4], linewidth=lw*2, color=colors[4], label='Iso, 700K')
#ax1.legend(loc='best', frameon=False, fontsize=ts)

ax2.set_ylabel('$\u03BA$(x) [W/(mK)]', fontsize=ts) 
ax2.set_xlabel('Distance from GB [$m$]', fontsize=ts)
ax2.set_title('$\gamma$ = 26.8', fontsize=ts)  
ax2.tick_params(axis="both", labelsize=ts-1)
ax2.grid(False)
ax2.xaxis.tick_bottom()
ax2.yaxis.tick_left()
ax2.set_xlim([1e-10, 1e-2])
ax2.set_ylim([-0.0005, 0.023])
ax2.set_xscale('log')

ax2.plot(x, ExRes_Fiso_tg_D[:,2,0], linewidth=lw*2, color=colors[0], label='Iso, 300K')
ax2.plot(x, ExRes_Fiso_tg_D[:,2,1], linewidth=lw*2, color=colors[1], label='Iso, 400K')
ax2.plot(x, ExRes_Fiso_tg_D[:,2,2], linewidth=lw*2, color=colors[2], label='Iso, 500K')
ax2.plot(x, ExRes_Fiso_tg_D[:,2,3], linewidth=lw*2, color=colors[3], label='Iso, 600K')
ax2.plot(x, ExRes_Fiso_tg_D[:,2,4], linewidth=lw*2, color=colors[4], label='Iso, 700K')
ax2.legend(loc='best', frameon=False, fontsize=ts, bbox_to_anchor=(1.05, 1))

fig.tight_layout()
name_of_figure = 'ExResx_tg_Iso_Temperature'
plt.savefig(output_path / name_of_figure, dpi=500, bbox_inches='tight')


# %%

''' PLOT - Excess resistivity profile X-component, transmission '''

fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(16,4))

ax.set_ylabel('$\u03BA$(x) [W/(mK)]', fontsize=ts) 
ax.set_xlabel('Distance from GB [$m$]', fontsize=ts) 
ax.set_title('$\gamma$ = 2.4', fontsize=ts) 
ax.tick_params(axis="both", labelsize=ts-1)
ax.grid(False)
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
ax.set_xlim([1e-10, 1e-2])
ax.set_ylim([-0.0005, 0.2])
ax.set_xscale('log')

ax.plot(x, ExRes_FX_tg_D[:,0,0], linewidth=lw*2, color=colors[0], label='X comp., 300K')
ax.plot(x, ExRes_FX_tg_D[:,0,1], linewidth=lw*2, color=colors[1], label='X comp., 400K')
ax.plot(x, ExRes_FX_tg_D[:,0,2], linewidth=lw*2, color=colors[2], label='X comp., 500K')
ax.plot(x, ExRes_FX_tg_D[:,0,3], linewidth=lw*2, color=colors[3], label='X comp., 600K')
ax.plot(x, ExRes_FX_tg_D[:,0,4], linewidth=lw*2, color=colors[4], label='X comp., 700K')
#ax.legend(loc='best', frameon=False, fontsize=ts)

ax1.set_ylabel('$\u03BA$(x) [W/(mK)]', fontsize=ts) 
ax1.set_xlabel('Distance from GB [$m$]', fontsize=ts) 
ax1.set_title('$\gamma$ = 14.6', fontsize=ts) 
ax1.tick_params(axis="both", labelsize=ts-1)
ax1.grid(False)
ax1.xaxis.tick_bottom()
ax1.yaxis.tick_left()
ax1.set_xlim([1e-10, 1e-2])
ax1.set_ylim([-0.0005, 0.2])
ax1.set_xscale('log')

ax1.plot(x, ExRes_FX_tg_D[:,1,0], linewidth=lw*2, color=colors[0], label='X comp., 300K')
ax1.plot(x, ExRes_FX_tg_D[:,1,1], linewidth=lw*2, color=colors[1], label='X comp., 400K')
ax1.plot(x, ExRes_FX_tg_D[:,1,2], linewidth=lw*2, color=colors[2], label='X comp., 500K')
ax1.plot(x, ExRes_FX_tg_D[:,1,3], linewidth=lw*2, color=colors[3], label='X comp., 600K')
ax1.plot(x, ExRes_FX_tg_D[:,1,4], linewidth=lw*2, color=colors[4], label='X comp., 700K')
#ax1.legend(loc='best', frameon=False, fontsize=ts)

ax2.set_ylabel('$\u03BA$(x) [W/(mK)]', fontsize=ts) 
ax2.set_xlabel('Distance from GB [$m$]', fontsize=ts)
ax2.set_title('$\gamma$ = 26.8', fontsize=ts)  
ax2.tick_params(axis="both", labelsize=ts-1)
ax2.grid(False)
ax2.xaxis.tick_bottom()
ax2.yaxis.tick_left()
ax2.set_xlim([1e-10, 1e-2])
ax2.set_ylim([-0.0005, 0.2])
ax2.set_xscale('log')

ax2.plot(x, ExRes_FX_tg_D[:,2,0], linewidth=lw*2, color=colors[0], label='X comp., 300K')
ax2.plot(x, ExRes_FX_tg_D[:,2,1], linewidth=lw*2, color=colors[1], label='X comp., 400K')
ax2.plot(x, ExRes_FX_tg_D[:,2,2], linewidth=lw*2, color=colors[2], label='X comp., 500K')
ax2.plot(x, ExRes_FX_tg_D[:,2,3], linewidth=lw*2, color=colors[3], label='X comp., 600K')
ax2.plot(x, ExRes_FX_tg_D[:,2,4], linewidth=lw*2, color=colors[4], label='X comp., 700K')
ax2.legend(loc='best', frameon=False, fontsize=ts, bbox_to_anchor=(1.05, 1))

fig.tight_layout()
name_of_figure = 'ExResx_tg_X_Debye_Temperature'
plt.savefig(output_path / name_of_figure, dpi=500, bbox_inches='tight')

# %% 

''' CALCULATION - EXCESS TBR, 300 K'''

gg = np.linspace(g[0], g[2], num=20) # finer range of gamma parameters

x_min = 0 # Define the limits for the integral over x
x_max = 1e-2 # [m]


# Excess TBR Perform the integral over x - Isotropic Supp Func
ExTBR_Fiso = np.zeros(len(gg))
for i, gi in enumerate(gg):
   ExTBR_Fiso[i], _ = quad(lambda x: excess_resistivity_Fiso_tg(x, gi), x_min, x_max, epsabs=1.49e-13, epsrel=1.49e-29)
ExTBC_Fiso = 1/ExTBR_Fiso/1e6 # Excess TBC [MW/(m2K)]


# Excess TBR Perform the integral over x - X-direction Supp Func
ExTBR_FX = np.zeros(len(gg))
for i, gi in enumerate(gg):
    ExTBR_FX[i], _ = quad(lambda x: excess_resistivity_FX_tg(x, gi, Debye=True), x_min, x_max, epsabs=1.49e-13, epsrel=1.49e-25)

ExTBC_FX = 1/ExTBR_FX/1e6 # MW/(m2K)

# %%

''' DEFINITION + CALCULATION - LANDAUER TBR, 300 K'''

# Landauer formalism Spectral thermal conductance

def Landauer_spectral_TBC_tg(w, g, T=300, Debye=True):
    if Debye: C = SpecHeat_Debye(w, T) 
    else: C = SpecHeat_HT(w)
    t = transmission(w, g)
    TBCspec = 1/4*t*C*v_Si # spectral thermal conductance
    return  TBCspec


# Perform the integral over w - Landauer formalism thermal conductance - frequency-dependent transmission, Debye Cp

TBC_Landauer = np.zeros(len(gg))
for i, gi in enumerate(gg):
    TBC_Landauer[i], _ = quad(lambda w: Landauer_spectral_TBC_tg(w, gi), w_min, w_max, epsabs=1.49e-12)
TBR_Landauer = 1/TBC_Landauer*1e9 # m2K/GW

# %%

''' PLOT - THERMAL BOUNDARY RESISTANCE 300 K '''

fig, (ax1) = plt.subplots(figsize=(5.3,4))

ax1.set_ylabel('$TBR_{Excess}$ [m$^2$K/GW]', fontsize=ts) 
ax1.set_xlabel('$\gamma$ $parameter$ [-]', fontsize=ts) 
ax1.tick_params(axis="both", labelsize=ts-1)
ax1.grid(False)
ax1.xaxis.tick_bottom()
ax1.yaxis.tick_left()
#ax1.set_xlim([0, 30])
ax1.set_ylim([-0.1, 10.1])

ax1.plot(gg, ExTBR_Fiso*1e9, linewidth=lw*2, color='k', label='Excess - Isotropic')
ax1.plot(gg, ExTBR_FX*1e9, linewidth=lw*2, color=colors[3], label='Excess - X-direction')
ax1.plot(gg, TBR_Landauer, linewidth=lw*2, color=colors[2], label='Landauer')

ax1.legend(loc='best', frameon=False, fontsize=ts-1)

fig.tight_layout()
name_of_figure = 'ExcessAndLandauerTBR'

# %%

''' CALCULATION - LANDAUER TBR, temperature-dependent '''

# Perform the integral over w - Landauer formalism thermal conductance - frequency-dependent transmission, Debye Cp
TBC_Landauer_T = np.zeros((len(g), len(T)))
TBR_Landauer_T = np.zeros((len(g), len(T)))

for j, Ti in enumerate(T):
    for i, gi in enumerate(g):
        TBC_Landauer_T[i,j], _ = quad(lambda w: Landauer_spectral_TBC_tg(w, gi, Ti), w_min, w_max, epsabs=1.49e-12)
        TBR_Landauer_T[i,j] = 1/TBC_Landauer_T[i,j]*1e9 # m2K/GW

  # %%   

''' CALCULATION - EXCESS TBR, temperature-dependent '''

# Excess TBR Perform the integral over x - Isotropic Supp Func
ExTBR_Temp_Fiso = np.zeros((len(g), len(T)))

for j in range(len(T)):
    for i in range(len(g)):
        ExTBR_Temp_Fiso[i,j] , _ = quad(lambda x: excess_resistivity_Fiso_tg(x, g[i], T[j]), x_min, x_max, epsabs=1.49e-13, epsrel=1.49e-25)


# Excess TBR Perform the integral over x - X-projected Supp Func
ExTBR_Temp_FX = np.zeros((len(g), len(T)))

# I am splitting the x range in 3 steps to allow a better and more accurate integration,
# that seems necessary for computing TBRs with the desired accuracy at higher temperatures
x_slot = np.array([0, 1e-8, 1e-5, 1e-2]) # [m]
Integration_slot = np.zeros((len(x_slot),len(g), len(T)))
error_slot = np.zeros((len(x_slot),len(g), len(T)))


for j in range(len(T)):
    for i in range(len(g)):
        for k in range(len(x_slot)-1):
            Integration_slot[k, i, j], error_slot[k, i, j] = quad(lambda x: excess_resistivity_FX_tg(x, g[i], T[j]), x_slot[k], x_slot[k+1], epsabs=1.49e-14, epsrel=1.49e-25)
        ExTBR_Temp_FX[i,j] = np.sum(Integration_slot[:, i, j])
   

# %%

''' PLOT - TBR, temperature-dependent '''

fig, (ax1) = plt.subplots(figsize=(8,4))

ax1.set_ylabel('TBR [m$^2$K/GW]', fontsize=ts) 
ax1.set_xlabel('Temperature [K]', fontsize=ts) 
ax1.tick_params(axis="both", labelsize=ts-1)
ax1.grid(False)
ax1.xaxis.tick_bottom()
ax1.yaxis.tick_left()
#ax1.set_xlim([0, 30])
#ax1.set_ylim([-0.1, 3.1])

ax1.plot(T, ExTBR_Temp_FX[0,:]*1e9, linewidth=lw*2, color=colors[0], label='$TBR_{Excess}$, $\gamma$ = 2.4')
ax1.plot(T, ExTBR_Temp_FX[1,:]*1e9, linewidth=lw*2, color=colors[1], label='$TBR_{Excess}$, $\gamma$ = 14.6')
ax1.plot(T, ExTBR_Temp_FX[2,:]*1e9, linewidth=lw*2, color=colors[2], label='$TBR_{Excess}$, $\gamma$ = 26.8')

ax1.plot(T, TBR_Landauer_T[0,:], '--', linewidth=lw*2, color=colors[0], label='$TBR_{Landauer}$, $\gamma$ = 2.4')
ax1.plot(T, TBR_Landauer_T[1,:], '--', linewidth=lw*2, color=colors[1], label='$TBR_{Landauer}$, $\gamma$ = 14.6')
ax1.plot(T, TBR_Landauer_T[2,:], '--', linewidth=lw*2, color=colors[2], label='$TBR_{Landauer}$, $\gamma$ = 26.8')

ax1.legend(loc='best', frameon=False, fontsize=ts-1, bbox_to_anchor=(1.05, 1))

fig.tight_layout()
name_of_figure = 'TBR_Temperature'
plt.savefig(output_path / name_of_figure, dpi=500, bbox_inches='tight', transparent=True)

# %%

