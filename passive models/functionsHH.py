#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:29:22 2020

@author: elisabeth
"""
import numpy as np
import math


import matplotlib.pyplot as plt


#from scipy.integrate import odeint
from scipy.integrate import solve_ivp

def calculateHH(gK=36.0,gNa=120.0,gL=0.3,Cm=1.0,VK=-12.0,VNa=115.0,Vl=10.613,tmin=0.0,tmax=50.0):
    """

    Parameters
    ----------
    gK : float, optional
        Average potassium channel conductance per unit area (mS/cm^2). The default is 36.0.
    gNa : float, optional
       Average sodium channel conductance per unit area (mS/cm^2). The default is 120.0.
    gL : float optional
        Average leak channel conductance per unit area (mS/cm^2). The default is 0.3.
    Cm : float, optional
        Membrane capacitance per unit area (uF/cm^2) The default is 1.0.
    VK : float, optional
        Potassium potential (mV) #nernst. The default is -12.0.
    VNa : float, optional
        Sodium potential (mV) #nernst. The default is 115.0.
    Vl : float, optional
        Leak potential (mV) #nernst. The default is 10.613.
    tmin : float, optional
        start time in ms. The default is 0.0.
    tmax : float, optional
        ens time in ms The default is 50.0.

    Returns
    -------
    None.

    """
     
        # Set random seed (for reproducibility)
    np.random.seed(1000)
    
    # Time values
    T = np.linspace(tmin, tmax, 10000)
    
    #use y as an array that contains all time dependent parameters, V,n,m,h
    # State (Vm, n, m, h) #set initial values
    Y = np.array([0.0, n_inf(), m_inf(), h_inf()])
    #formulate derivatives
    #dy=formulate_derivatives(Y, T[0],gK,gNa,gL,Cm,VK,VNa,Vl)
    
    # Solve ODE system
    # Vy = (Vm[t0:tmax], n[t0:tmax], m[t0:tmax], h[t0:tmax])
    #Vy = odeint(compute_derivatives, Y, T)
    Vy=solve_ivp(compute_derivatives, [tmin,tmax], Y,t_eval=T,args=(Y, T[0],gK,gNa,gL,Cm,VK,VNa,Vl))
    #Vy.t gives back T
    #Vy.y gives back the solution
    
    # Input stimulus
    Idv = [Id(t) for t in T]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(T, Idv)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(r'Current density (uA/$cm^2$)')
    ax.set_title('Stimulus (Current density)')
    plt.grid()
    
    # Neuron potential
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(T, Vy[:, 0])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Vm (mV)')
    ax.set_title('Neuron potential with two spikes')
    plt.grid()
    X=1
    
    
    return X


"""
All the used functions are down below
"""
def compute_derivatives(y, t0,gK,gNa,gL,Cm,VK,VNa,Vl):
        dy = np.zeros((4,))
        
        Vm = y[0]
        n = y[1]
        m = y[2]
        h = y[3]
        
        # dVm/dt
        GK = (gK / Cm) * np.power(n, 4.0)
        GNa = (gNa / Cm) * np.power(m, 3.0) * h
        GL = gL / Cm
        
        dy[0] = (Id(t0) / Cm) - (GK * (Vm - VK)) - (GNa * (Vm - VNa)) - (GL * (Vm - Vl))
        
        # dn/dt
        dy[1] = (alpha_n(Vm) * (1.0 - n)) - (beta_n(Vm) * n)
        
        # dm/dt
        dy[2] = (alpha_m(Vm) * (1.0 - m)) - (beta_m(Vm) * m)
        
        # dh/dt
        dy[3] = (alpha_h(Vm) * (1.0 - h)) - (beta_h(Vm) * h)
        
        return dy

# Potassium ion-channel rate functions

def alpha_n(Vm):
    return (0.01 * (10.0 - Vm)) / (np.exp(1.0 - (0.1 * Vm)) - 1.0)

def beta_n(Vm):
    return 0.125 * np.exp(-Vm / 80.0)

# Sodium ion-channel rate functions

def alpha_m(Vm):
    return (0.1 * (25.0 - Vm)) / (np.exp(2.5 - (0.1 * Vm)) - 1.0)

def beta_m(Vm):
    return 4.0 * np.exp(-Vm / 18.0)

def alpha_h(Vm):
    return 0.07 * np.exp(-Vm / 20.0)

def beta_h(Vm):
    return 1.0 / (np.exp(3.0 - (0.1 * Vm)) + 1.0)
  
# n, m, and h steady-state values

def n_inf(Vm=0.0):
    return alpha_n(Vm) / (alpha_n(Vm) + beta_n(Vm))

def m_inf(Vm=0.0):
    return alpha_m(Vm) / (alpha_m(Vm) + beta_m(Vm))

def h_inf(Vm=0.0):
    return alpha_h(Vm) / (alpha_h(Vm) + beta_h(Vm))





# Input stimulus
def Id(t):
    'this is an example stimulation where a 150 mA pulse is given at t[0,1] and 50 ma btw t[10,11]'
    if 0.0 < t < 1.0:
        return 150.0
    elif 10.0 < t < 11.0:
        return 50.0
    return 0.0